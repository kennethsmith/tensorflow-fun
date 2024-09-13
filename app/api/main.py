from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import os

import app.api.tf.fashion_mnist.model as fm
import app.api.tf.cifar_10.l4.model as ct_l4
import app.api.tf.cifar_10.rnin.model as ct_rnin

app = FastAPI()


class Item(BaseModel):
    name: Union[str, None] = None
    url: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.get("/fashion_mnist/predict")
def read_prediction(item: Item):
    r = fm.predict(item.url, os.getcwd() + '/app/api/tf/fashion_mnist/')
    return {"result": r}


@app.get("/cifar_10/l4/predict")
def read_prediction(item: Item):
    r = ct_l4.predict(item.url, os.getcwd() + '/app/api/tf/cifar_10/l4/')
    return {"result": r}


@app.get("/cifar_10/rnin/predict")
def read_prediction(item: Item):
    r = ct_rnin.predict(item.url, os.getcwd() + '/app/api/tf/cifar_10/rnin/')
    return {"result": r}