from helper_functions import *

if __name__ == '__main__':
    def test_convert_to_forward_slashes():
        # Test case 1: Path with backslashes
        path1 = r'C:\Users\navme\Desktop\ECG_Thesis_Local\CODE-15 Datasets\Dataset\exams_part0.hdf5'
        expected_output1 = 'C:/Users/navme/Desktop/ECG_Thesis_Local/CODE-15 Datasets/Dataset/exams_part0.hdf5'
        assert convert_to_forward_slashes(path1) == expected_output1
        
        # Test case 2: Path with forward slashes
        path2 = '/Users/navme/Desktop/ECG_Thesis_Local/CODE-15 Datasets/Dataset/exams_part0.hdf5'
        expected_output2 = '/Users/navme/Desktop/ECG_Thesis_Local/CODE-15 Datasets/Dataset/exams_part0.hdf5'
        assert convert_to_forward_slashes(path2) == expected_output2
        
        # Test case 3: Path with mixed slashes
        path3 = r'C:\Users\navme\Desktop/ECG_Thesis_Local/CODE-15 Datasets/Dataset/exams_part0.hdf5'
        expected_output3 = 'C:/Users/navme/Desktop/ECG_Thesis_Local/CODE-15 Datasets/Dataset/exams_part0.hdf5'
        assert convert_to_forward_slashes(path3) == expected_output3
        
        # Test case 4: Path with no slashes
        path4 = 'exams_part0.hdf5'
        expected_output4 = 'exams_part0.hdf5'
        assert convert_to_forward_slashes(path4) == expected_output4
        
        print("All test cases passed!")
    
    test_convert_to_forward_slashes()