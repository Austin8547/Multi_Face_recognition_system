
import cv2
import numpy as np
import onnxruntime as ort


class ArcFaceONNX:
    """ArcFace ONNX wrapper with batched CUDA inference."""

    def __init__(self, model_path):
        providers = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("ArcFace providers:", self.session.get_providers())
        print("Input shape:", self.session.get_inputs()[0].shape)
        print("Output shape:", self.session.get_outputs()[0].shape)

    def _preprocess_batch(self, aligned_faces):
        """
        Input: list of BGR aligned faces, each 112x112x3
        Output: float32 tensor (N, 3, 112, 112)
        """

        if not aligned_faces:
            return np.empty((0, 3, 112, 112), dtype=np.float32)

        # Stack faces into one batch
        batch = np.stack(aligned_faces, axis=0)

        # BGR -> RGB, normalize and convert to NCHW
        batch = batch[..., ::-1].astype(np.float32)
        batch = (batch - 127.5) / 128.0
        batch = batch.transpose(0, 3, 1, 2)

        return np.ascontiguousarray(batch, dtype=np.float32)

    def get_embeddings(self, aligned_faces):
        """
        Run one ONNX inference call for all faces.

        Returns: (N, 512) L2-normalized embeddings.
        """

        if not aligned_faces:
            return np.empty((0, 512), dtype=np.float32)

        batch = self._preprocess_batch(aligned_faces)

        embeddings = self.session.run(
            [self.output_name],
            {self.input_name: batch}
        )[0]

        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = embeddings.reshape(len(aligned_faces), -1)

        # Normalize each face embedding independently
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-6)

        return embeddings

    def get_embedding(self, aligned_face):
        """Backward-compatible single-face method."""

        return self.get_embeddings([aligned_face])[0]
