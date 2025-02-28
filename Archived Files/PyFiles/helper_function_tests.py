from helper_functions import *
from tripletloss import TripletLoss
import torch

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

    def test_triplet_loss():
        # Create some random input tensors
        anchor = torch.randn(4, 20)
        positive = torch.randn(4, 20)
        negative = torch.randn(4, 20)

        # Create an instance of the TripletLoss class
        triplet_loss = TripletLoss(margin=1.0)

        # Compute the loss using the forward method of the TripletLoss class
        loss = triplet_loss(anchor, positive, negative)
        print(loss)

        # Compute the expected loss using the formula for the Triplet Loss
        expected_loss = torch.mean(torch.clamp(F.pairwise_distance(anchor, positive, keepdim=True) - F.pairwise_distance(anchor, negative, keepdim=True) + 1.0, min=0.0))
        print(expected_loss)

        # Check that the computed loss is equal to the expected loss
        assert torch.allclose(loss, expected_loss)
        print(torch.allclose(loss, expected_loss))

