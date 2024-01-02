from attenuation.datasets import wikitext2
# from attenuation.datasets import IWSLT2017

DATASETS = {
    wikitext2.DATASET_NAME: wikitext2.load,
    # IWSLT2017.DATASET_NAME: IWSLT2017.load,
}
