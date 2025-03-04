from templates_cls import *
from experiment_classifier import *
from sklearn import metrics

if __name__ == '__main__':
    # need to first train the diffae autoencoding model & infer the latents
    # this requires only a single GPU.
    gpus = [0]
    conf = ffhq128_autoenc_cls()
    model = train_cls(conf, gpus=gpus)

    # after this you can do the manipulation! model = ClsModel(conf)
    test_dataset = CelebHQAttrDataset(data_paths['celebahq'],
                                    conf.img_size,
                                    data_paths['celebahq_anno'],
                                    do_augment=False, mode='test')
    loader = conf.make_loader(test_dataset, shuffle=False, drop_last=True)
    predictions = []
    all_labels = []
    conf.batch_size = 512
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            imgs = batch['img']
            labels = batch['labels']
            cond = model.ema_model.encoder(imgs)
            pred = model.classifier.forward(cond)
            predictions.append(pred.cpu())
            all_labels.append(labels.cpu())
    predictions = torch.cat(predictions)
    all_labels = torch.cat(all_labels)
    torch.save(predictions, "predictions_bs.pt")
    torch.save(all_labels, "all_labels_bs.pt")
    all_labels[torch.where(all_labels == -1)] = 0
    print("overall acc:", torch.sum((predictions > 0) == all_labels) / len(predictions.flatten()))
    
    aucs = []
    positive_samples = []
    for i in range(40):
        cur_labels = all_labels[:, i]
        cur_predictions = predictions[:, i]
        fpr, tpr, thresholds = metrics.roc_curve(cur_labels, cur_predictions.float(), pos_label=1)
        aucs.append(metrics.auc(fpr, tpr))
        positive_samples.append(torch.sum(cur_labels).item())
    print(aucs)
    print(positive_samples)
    print("mean:", np.mean(aucs))
    print("weighted average:", np.sum((np.array(aucs) * np.array(positive_samples))) / np.sum(positive_samples))