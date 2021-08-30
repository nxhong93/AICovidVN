import __init__
import numpy as np
from glob import glob
import timm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from timm.optim.adabelief import AdaBelief
from timm.optim.radam import RAdam
from timm.optim.lookahead import Lookahead
from dataset import CovidDataset
from config.configs import *
from bce_loss import *
from network import *
from madgrad import *
from warmup_scheduler import *
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import AUROC


class covidNet(pl.LightningModule):
    def __init__(self, config, meta_col, df, test_df, fold=0, is_train=True, oof=False, is_sub=True):
        super(covidNet, self).__init__()

        self.config = config
        self.fold = fold
        self.df = df
        self.test_df = test_df
        self.is_train = is_train
        self.meta_col = meta_col
        self.oof = oof
        self.is_sub = is_sub


        self.backbond = timm.create_model(config.model_name,
                                          pretrained=is_train,
                                          in_chans=1)
        if hasattr(self.backbond, 'fc'):
            in_features = self.backbond.fc.in_features
            if self.config.use_gem:
                self.backbond.global_pool = GeM()
            self.backbond.fc = nn.Identity()
        elif hasattr(self.backbond, 'head'):
            if hasattr(self.backbond.head, 'fc'):
                in_features = self.backbond.head.fc.in_features
                if self.config.use_gem:
                    self.backbond.head.fc.global_pool = GeM()
                self.backbond.head.fc = nn.Identity()
            else:
                in_features = self.backbond.head.in_features
                if self.config.use_gem:
                    self.backbond.head.global_pool = GeM()
                self.backbond.head = nn.Identity()
        else:
            in_features = self.backbond.classifier.in_features
            if self.config.use_gem:
                self.backbond.global_pool = GeM()
            self.backbond.classifier = nn.Identity()

        self.mfcc_conv = MfccNet(in_channels=1, out_channels=64)
        if self.config.useMeta:
            self.meta_linear = nn.Linear(nn.Linear(len(meta_col), 8),
                                         nn.BatchNorm1d(8),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Linear(8, 64),
                                         nn.BatchNorm1d(64))

            self.fc = nn.Sequential(nn.Linear(in_features + 64 * 2, 64),
                                    nn.BatchNorm1d(64),
                                    nn.Dropout(0.5),
                                    nn.Linear(64, 2))
        else:
            self.fc = nn.Sequential(nn.Linear(in_features + 64, 64),
                                    nn.BatchNorm1d(64),
                                    nn.Dropout(0.5),
                                    nn.Linear(64, 2))

        if self.config.use_weight:
            label_inv = (1 / self.df["assessment_result"].value_counts().sort_index()).values
            label_inv_mean = label_inv.mean()
            weight = label_inv * (1 / label_inv_mean)
            weight = torch.tensor(weight).to('cuda')
        else:
            weight = None
        self.loss_fn = criterionLoss(use_focal=self.config.use_focal, weight=weight)

    def forward(self, mf, mel, meta):
        mel = self.backbond(mel).squeeze(-1).squeeze(-1)
        mf = self.mfcc_conv(mf)

        if self.config.useMeta:
            meta = self.meta_linear(meta)
            x = torch.cat([mf, mel, meta], axis=-1)
        else:
            x = torch.cat([mf, mel], axis=-1)
        x = self.fc(x)
        return torch.sigmoid(x)

    def configure_optimizers(self):
        if self.config.optimizer == 'sam':
            optimizer = SAMSGD(self.parameters(), lr=self.config.lr,
                               momentum=self.config.momentum)
        elif self.config.optimizer == 'madgrad':
            optimizer = MADGRAD(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'adam':
            optimizer = AdaBelief(self.parameters(), lr=self.config.lr)
        elif self.config.optimizer == 'ranger':
            base_optimizer = RAdam(self.parameters(), lr=self.config.lr)
            optimizer = Lookahead(base_optimizer, k=6, alpha=0.5)
        scheduler = self.config.SchedulerCosine(optimizer, **self.config.cosine_params)
        if self.config.has_warmup:
            scheduler = GradualWarmupSchedulerV2(optimizer, after_scheduler=scheduler,
                                                 **self.config.warmup_params)

        return [optimizer], [scheduler]

    def prepare_data(self):
        train_data = self.df[self.df['fold'] != self.fold].reset_index(drop=True)
        val_data = self.df[self.df['fold'] == self.fold].reset_index(drop=True)
        # Create dataset
        self.train_ds = CovidDataset(train_data, DatasetConfig, meta_col=self.meta_col, sub='train', has_transform=True)
        self.valid_ds = CovidDataset(val_data, DatasetConfig, meta_col=self.meta_col, sub='validation', has_transform=False)
        self.test_ds = CovidDataset(self.test_df, DatasetConfig, meta_col=self.meta_col, sub='test', has_transform=True)

    def train_dataloader(self):
        loader = DataLoader(self.train_ds, batch_size=self.config.batch_size, shuffle=True,
                            pin_memory=True, num_workers=self.config.num_workers,
                            collate_fn=self.train_ds.collate_fn, drop_last=True)

        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valid_ds, batch_size=self.config.batch_size, shuffle=False,
                            pin_memory=True, num_workers=self.config.num_workers,
                            collate_fn=self.valid_ds.collate_fn, drop_last=True)

        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_ds, batch_size=self.config.batch_size, shuffle=False,
                            pin_memory=True, num_workers=self.config.num_workers,
                            collate_fn=self.test_ds.collate_fn)

        return loader

    def training_step(self, batch, batch_idx):
        self.optimizers()
        mf, mel, meta, label = batch
        if self.config.alpha is not None:
            mel, label0, label1, lam = mixup_data(mel, label, alpha=self.config.alpha)
            output = self(mf, mel, meta)
            loss = mixup_criterion(self.loss_fn, output, label0, label1, lam)
        else:
            output = self(mf, mel, meta)
            loss = self.loss_fn(output, label)

        self.log('loss', loss, prog_bar=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        mf, mel, meta, label = batch
        output = self(mf, mel, meta)
        loss = self.loss_fn(output, label)
        self.log("val_loss", loss, prog_bar=False)
        return {'val_loss': loss, 'label': label[:, 1].detach().cpu().numpy(),
                'out': output[:, 1].squeeze().detach().cpu().numpy()}

    def test_step(self, batch, batch_idx):
        mf, mel, meta = batch
        pred = self(mf, mel, meta)
        return {'pred': pred[:, 1].squeeze().detach().cpu().numpy()}

    def validation_epoch_end(self, output):
        label = np.concatenate([x['label'] for x in output])
        out = np.concatenate([x['out'] for x in output])
        accuracy = accuracy_score(label, np.where(out >= 0.5, 1, 0))
        auc = roc_auc_score(label, out)
        avg_loss = torch.stack([x['val_loss'] for x in output]).mean()
        score = 0.5 * (auc + accuracy)
        self.log('accuracy', accuracy, prog_bar=False)
        self.log('auc', auc, prog_bar=False)
        self.log('score', score, prog_bar=False)
        print(f'Validation: loss {avg_loss:.5f} | accuracy: {accuracy:.5f} | auc: {auc:.5f} | score: {score:.5f}')

    def test_epoch_end(self, output):
        pred = np.concatenate([np.atleast_1d(x['pred']) for x in output])
        if self.is_sub:
            if self.oof:
                self.test_df.loc[:, f'target_fold{self.fold}'] = pred
                self.test_df[['uuid', f'target_fold{self.fold}']] \
                    .to_csv(f'oof_fold{self.fold}.csv', index=False)
                return {'oof_fold': self.fold}
            else:
                N = len(glob(f'submission_fold{self.fold}_*.csv'))
                self.test_df.loc[:, f'target_fold{self.fold}_{N}'] = pred
                self.test_df[['uuid', f'target_fold{self.fold}_{N}']] \
                    .to_csv(f'submission_fold{self.fold}_{N}.csv', index=False)
                return {'tta': N}
        else:
            self.pred = pred
            return pred

