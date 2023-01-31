## Scripts

The main entry spots to the code is `infer_group_label.py` for inferring group label, and `gdro/group_dro.py` for training debiased classifier.

Make sure to run following code under this directory `b2t/b2t_debias`.

### Inferring Group Labels

To reproduce the results corresponding to Table 3 in our paper, run following script.

```bash
$ python infer_group_label.py --data-dir [PATH TO DATASET] --dataset [DATASET] --save_path pseudo_bias/[DATASET].pt
```

###  Training debiased classifiers

To train debiased classifiers with GroupDRO using ground truth group label, run following script. 

```bash
$ bash gdro/scripts/run_dro_[DATASET].sh [PATH TO DATASET] [SEED]
```

To train debiased classifier using B2T inferred group labels, run following script. 

```bash
$ bash gdro/scripts/run_dro_[DATASET]_b2t.sh [PATH TO DATASET] [SEED]
```

We used the official [GroupDRO](https://github.com/kohpangwei/group_DRO) implementation on Waterbirds dataset. 