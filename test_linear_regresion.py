# test_linear_regresion.py
import unittest
import numpy as np
from linear_regresion import linear_regresion

class TestPandasExercises(unittest.TestCase):
    def test_linear_regresion(self):
        mse, r2 = linear_regresion()
        ok = np.isclose(mse, 4198.86, rtol=1.0) and np.isclose(r2, 0.23, rtol=0.2)
        self.assertTrue(ok, msg=f"Esperado ~ (4198.86, 0.23), se obtuvo (mse={mse:.2f}, r2={r2:.4f})")

if __name__ == "__main__":
    unittest.main()
