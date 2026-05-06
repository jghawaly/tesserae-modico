"""Baseline file fragment classifiers for comparison with MoDiCo."""

from .bytercnn import ByteRCNN, create_bytercnn_model
from .cnn_lstm import CNNLSTM, create_cnn_lstm_model
from .dscse import DSCSE, create_dscse_model
from .fifty import FiFTy, create_fifty_model

__all__ = [
    "ByteRCNN",
    "create_bytercnn_model",
    "FiFTy",
    "create_fifty_model",
    "CNNLSTM",
    "create_cnn_lstm_model",
    "DSCSE",
    "create_dscse_model",
]
