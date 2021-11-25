import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import DDPPlugin

#from RACEDataModule import RACEDataModule
from ALBERTForRace import ALBERTForRace
from RACEMemDataModule import RACEMemDataModule
from ALBERTForRaceMem import ALBERTForRaceMem

if __name__ == '__main__':
    model = ALBERTForRaceMem(
        pretrained_model='albert-base-v2',
        learning_rate=2e-5,
        num_train_epochs=2,
        train_batch_size=1,
        train_all=True,
        use_bert_adam=True,
    )
    dm = RACEMemDataModule(
        model_name_or_path='albert-base-v2',
        datasets_loader='race',
        train_batch_size=1,
        max_seq_length=160,
        num_workers=8,
        num_preprocess_processes=48,
    )
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None, # ！！！-1 出现 out of mem, 1 or 4 nan problem
        amp_backend='native',
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        gradient_clip_val=1.0,
        #accumulate_grad_batches=2,
        max_epochs=10,
        plugins=DDPPlugin(find_unused_parameters=True),
        val_check_interval=0.2,
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
