import unittest
import src.modules.database as db

class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.db = db.Database()

    def test_select_image(self):
        print("Validate test_select_image:", self.db.select_image(2, 2))


if __name__ == '__main__':
    unittest.main()