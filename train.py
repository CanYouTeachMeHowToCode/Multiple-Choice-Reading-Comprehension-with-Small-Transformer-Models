import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import DDPPlugin

from RACEDataModule import RACEDataModule
from ALBERTForRace import ALBERTForRace


if __name__ == '__main__':
    model = ALBERTForRace(
        pretrained_model='albert-base-v2',
        learning_rate=2e-5,
        num_train_epochs=2,
        train_batch_size=4,
        train_all=True,
        use_bert_adam=True,
    )
    dm = RACEDataModule(
        model_name_or_path='albert-base-v2',
        datasets_loader='race',
        train_batch_size=4,
        max_seq_length=384,
        num_workers=8,
        num_preprocess_processes=48,
    )
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        gradient_clip_val=1.0,
        max_epochs=10,
        plugins=DDPPlugin(find_unused_parameters=True),
        val_check_interval=0.2,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
