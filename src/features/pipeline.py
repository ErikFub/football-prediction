from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


preprocessing_pipeline = Pipeline(
    steps=[('imputer', SimpleImputer(strategy='mean')),
           ('scaler', StandardScaler())]
)


def construct_full_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=preprocessing_pipeline.steps + [('predictor', estimator)]
    )
