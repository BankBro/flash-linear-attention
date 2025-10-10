from lightning.pytorch import LightningModule, LightningDataModule
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging

# 导入日志工具
from train.utils.logger_utils import logger


class SeqTaskModule(LightningModule):
    def __init__(self, cfg, vocab_size):
        super().__init__()

        if not isinstance(cfg, DictConfig):
            raise TypeError("Config must be an instance of DictConfig.")
        
        # 确保checkpoint内自动包含checkpoint['hyper_parameters']
        # 同时self.hparams可以直接访问config
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        logger.info(f"Instantiating model <{self.cfg.model._target_}>")
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            vocab_size = vocab_size,
            task_cfg = cfg,
            _recursive_ = False)
        logger.info(f"Model structure:\n{self.model}")

        logger.info(f"Instantiating loss_fn <{self.cfg.loss_fn._target_}>")
        self.loss_fn = hydra.utils.instantiate(self.cfg.loss_fn, _recursive_=False)

        # self.kan_loss_weight = self.cfg.get("kan_loss_weight", 0.0)
        self.kan_loss_weight = 1.0

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch, phase):
        input, target = batch
        outputs = self(input)

        outputs = outputs.view(-1, outputs.size(-1))  # [B*S, V]
        target = target.view(-1)  # [B*S]

        # 主要损失
        main_loss = self.loss_fn(outputs, target)
        ppl = torch.exp(main_loss)

        # KAN损失
        kan_loss = self._collect_kan_loss(phase)

        # 训练阶段：只记录到logger，不在终端显示（由自定义callback处理）
        # 验证和测试阶段：在epoch结束时记录
        if phase == "train":
            total_loss = main_loss + self.kan_loss_weight * kan_loss
            # print(f"Phase: {phase}, Main Loss: {main_loss:.4f}, KAN Loss: {kan_loss:.4f}, Total Loss: {total_loss:.4f}, PPL: {ppl:.2f}")

            # 只记录到logger（如wandb），不在终端显示进度条  MY TODO: 提供一个开关选择显示bar还是logger
            self.log(f"{phase}/main_loss", main_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f"{phase}/kan_loss", kan_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f"{phase}/loss", total_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f"{phase}/ppl", ppl, on_step=True, on_epoch=False, prog_bar=False, logger=True)

            return {"loss": total_loss, "total_loss": total_loss, "main_loss": main_loss, "kan_loss": kan_loss, "ppl": ppl}

        else:
            # 验证和测试时，在epoch结束时记录
            self.log(f"{phase}/loss", main_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log(f"{phase}/ppl", ppl, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return {"loss": main_loss, "ppl": ppl}
    
    def _collect_kan_loss(self, phase):
        total_kan_loss = 0.0

        if phase != "train":
            return 0.0

        for layer in self.model.model.model.layers:
            layer_idx = layer.layer_idx if hasattr(layer, 'layer_idx') else 'unknown'

            layer_q_kan_loss = layer.attn.q_kan_activation.current_loss if hasattr(layer.attn, 'q_kan_activation') else 0.0
            layer_k_kan_loss = layer.attn.k_kan_activation.current_loss if hasattr(layer.attn, 'k_kan_activation') else 0.0
            total_kan_loss += layer_q_kan_loss + layer_k_kan_loss

            layer_q_aux_loss = layer.attn.q_kan_activation.aux_loss if hasattr(layer.attn, 'q_kan_activation') else 0.0
            layer_k_aux_loss = layer.attn.k_kan_activation.aux_loss if hasattr(layer.attn, 'k_kan_activation') else 0.0
            total_kan_loss += layer_q_aux_loss + layer_k_aux_loss

            self.log(f"{phase}/q_kan_loss/layer_{layer_idx}", layer_q_kan_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f"{phase}/k_kan_loss/layer_{layer_idx}", layer_k_kan_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f"{phase}/q_aux_loss/layer_{layer_idx}", layer_q_aux_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log(f"{phase}/k_aux_loss/layer_{layer_idx}", layer_k_aux_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

            total_kan_loss += layer.attn.q_kan_activation.convex_concave_loss
            total_kan_loss += layer.attn.k_kan_activation.convex_concave_loss

            # logger.info(f"Layer {layer_idx} - "
            #             f"Q KAN Loss: {layer_q_kan_loss:.4f}, "
            #             f"K KAN Loss: {layer_k_kan_loss:.4f}, "
            #             f"Q Aux Loss: {layer_q_aux_loss:.4f}, "
            #             f"K Aux Loss: {layer_k_aux_loss:.4f}, "
            #             f"Q Convex/Concave Loss: {layer.attn.q_kan_activation.convex_concave_loss:.4f}, "
            #             f"K Convex/Concave Loss: {layer.attn.k_kan_activation.convex_concave_loss:.4f}")
        
        return total_kan_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers_no_group(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer,
            params=self.parameters(),
            _recursive_=False
        )
        print(f"Instantiating optimizer <{self.cfg.optimizer._target_}>")

        if not "lr_scheduler" in self.cfg:
            return optimizer

        lr_scheduler = hydra.utils.instantiate(
            self.cfg.lr_scheduler.timm_cosine,
            optimizer=optimizer,
            _recursive_=False
        )
        print(f"Instantiating lr_scheduler <{self.cfg.lr_scheduler.timm_cosine._target_}>")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': lr_scheduler,
                'interval': self.cfg.lr_scheduler.get('scheduler_interval', 'step'),
                'monitor': self.cfg.lr_scheduler.get('scheduler_monitor', 'val/loss')
            }
        }

    def configure_optimizers_group(self):
        # 分离KAN参数和其他参数
        kan_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if 'spline_weight' in name:
                kan_params.append(param)
                logger.debug(f"Found KAN parameter: {name}, shape: {param.shape}")
            else:
                other_params.append(param)
        
        logger.info(f"KAN parameters: {len(kan_params)}, Other parameters: {len(other_params)}")
        
        if len(kan_params) == 0:
            logger.info("No KAN parameters found, using single optimizer")
            # 如果没有KAN参数，使用原来的单优化器方式
            optimizer = hydra.utils.instantiate(
                self.cfg.optimizer,
                params=self.parameters(),
                _recursive_=False
            )
            logger.info(f"Instantiating optimizer <{self.cfg.optimizer._target_}>")
            
            if "lr_scheduler" not in self.cfg:
                return optimizer
                
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.lr_scheduler.timm_cosine,
                optimizer=optimizer,
                _recursive_=False
            )
            logger.info(f"Instantiating lr_scheduler <{self.cfg.lr_scheduler.timm_cosine._target_}>")
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": self.cfg.lr_scheduler.get('scheduler_interval', 'step'),
                    "monitor": self.cfg.lr_scheduler.get('scheduler_monitor', 'val/loss'),
                    "frequency": 1,
                    "name": "main_lr"
                }
            }
        
        # 使用参数组方式创建单个优化器
        base_lr = self.cfg.optimizer.lr
        kan_lr_multiplier = 5.0
        kan_lr = base_lr * kan_lr_multiplier
        
        # 创建参数组
        param_groups = [
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': self.cfg.optimizer.weight_decay
            },
            {
                'params': kan_params,
                'lr': kan_lr,
                'weight_decay': 0.0  # KAN参数不使用权重衰减
            }
        ]
        
        # 创建单个优化器但包含多个参数组
        optimizer_cfg = self.cfg.optimizer.copy()
        optimizer_cfg.pop('lr', None)  # 移除lr，因为在param_groups中指定了
        
        optimizer = hydra.utils.instantiate(
            optimizer_cfg,
            params=param_groups,
            # params=self.parameters(),
            _convert_='all',
            _recursive_=False
        )
        
        logger.info(f"Instantiating optimizer <{self.cfg.optimizer._target_}>")
        logger.info(f"Parameter group 0 (other): LR={base_lr}, weight_decay={self.cfg.optimizer.weight_decay}")
        logger.info(f"Parameter group 1 (KAN): LR={kan_lr}, weight_decay=0.0")
        
        if "lr_scheduler" not in self.cfg:
            return optimizer
        
        # 创建学习率调度器，它会自动处理多个参数组
        # 注意：调度器的固定参数（lr_min, warmup_lr_init）会应用到所有参数组
        lr_scheduler = hydra.utils.instantiate(
            self.cfg.lr_scheduler.timm_cosine,
            optimizer=optimizer,
            _recursive_=False
        )
        
        logger.info(f"Instantiating lr_scheduler <{self.cfg.lr_scheduler.timm_cosine._target_}>")
        logger.info(f"Note: lr_min={self.cfg.lr_scheduler.timm_cosine.lr_min} and "
                    f"warmup_lr_init={self.cfg.lr_scheduler.timm_cosine.warmup_lr_init} "
                    f"will be applied to all parameter groups")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.cfg.lr_scheduler.get('scheduler_interval', 'step'),
                "monitor": self.cfg.lr_scheduler.get('scheduler_monitor', 'val/loss'),
                "frequency": 1,
                "name": "main_lr"
            }
        }

    def configure_optimizers(self):
        lr_amplifier_flag = self.cfg.expt_params.get('lr_amplifier', False)

        if lr_amplifier_flag:
            return self.configure_optimizers_group()
        else:
            return self.configure_optimizers_no_group()