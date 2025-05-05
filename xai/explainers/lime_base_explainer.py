import PIL, sklearn
import numpy as np
from copy import deepcopy
from functools import partial
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from utils import get_batch_predict_function

class LimeBaseExplainer(object):
    def __init__(self, model_type, model, model_regressor, mean, std, num_samples, batch_size=None, kernel_width=0.25):
        self.model_type = model_type
        self.model = model
        self.mean = mean
        self.std = std
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.kernel_width = kernel_width
        
        self.model_regressor = None
        if model_regressor == "LinReg": self.model_regressor = LinearRegression(fit_intercept=True, n_jobs=-1)
        elif model_regressor == "Ridge": self.model_regressor = Ridge(alpha=1.0, fit_intercept=True)
        elif model_regressor == "Lasso": self.model_regressor = Lasso(alpha=1.0, fit_intercept=True)
        elif model_regressor == "ElasticNet": self.model_regressor = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True)

        def kernel(d, kernel_width): return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        self.kernel_fn = partial(kernel, kernel_width=self.kernel_width)

        self.classifier_fn = get_batch_predict_function(self.model_type)
    
    def explain_instance(self, image, segments, label, distance_metric='euclidean'):
        if type(image) == PIL.Image.Image: image = np.array(image)
        
        fudged_image = image.copy()
        mean_int = [255*m for m in self.mean]
        fudged_image[:] = mean_int

        data, labels, n_maskings = self.data_labels(image, fudged_image, segments)
        
        distances = sklearn.metrics.pairwise_distances(data, data[0].reshape(1, -1), metric=distance_metric).ravel()
        weights = self.kernel_fn(distances)
        labels_column = labels[:, label]

        self.model_regressor.fit(data, labels_column, weights)
        sp_names = np.unique(segments)

        attr_scores = dict()
        for sp_name, sp_attr in zip(sp_names, self.model_regressor.coef_):
            attr_scores[sp_name] = float(sp_attr)
        
        return attr_scores, n_maskings, data

    def data_labels(self, image, fudged_image, segments):
        n_features = np.unique(segments).shape[0]

        data = np.random.randint(0, 2, self.num_samples*n_features).reshape((self.num_samples, n_features))
        data[0, :] = 1

        samples, preds = list(), list()
        n_maskings = list()
        sp_names = np.unique(segments)
        for row in data:
            sample = deepcopy(image)
            sp_idxs_to_zero = np.where(row==0)[0]
            n_maskings.append(len(sp_idxs_to_zero))
            mask = np.zeros(segments.shape).astype(bool)
            for idx in sp_idxs_to_zero: mask[np.where(segments == sp_names[idx])] = True
            sample[mask] = fudged_image[mask]
            samples.append(sample)
            
            if self.batch_size is not None and len(samples) == self.batch_size:
                batch_preds = self.classifier_fn(self.model, samples, self.mean, self.std)
                preds.extend(batch_preds.tolist())
                samples = list()
        
        if self.batch_size is not None and len(samples) > 0:
            final_batch_preds = self.classifier_fn(self.model, samples, self.mean, self.std)
            preds.extend(final_batch_preds.tolist())
        
        elif self.batch_size is None:
            preds = self.classifier_fn(self.model, samples, self.mean, self.std).tolist()

        return data, np.array(preds), n_maskings