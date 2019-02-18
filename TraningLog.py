# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:57:11 2019

@author: shihao

Trianing log for CNN
"""

"""
01 26 2019
3pm:
    
    number of epoch changed from 20 to 14 to get minimal Validation Loss:
        Not much difference, Imaginary part is off

3:04pm:
    
    number of epoch changed back to 20, add two drop out layers:
        BAD, definitely no dropout
        
3:13pm:
    
    number of nuerons in the 1st fully connected layer changed to 256:
        better results, number of nuerons more the better

3.40pm:
    
    add another convolutional layer into the network:
        better results, considering adding another layer into it
        
3:51pm:
    
    add number of filters into the 1st convolutional layer
    from 32 to 64:
        better results, slightly

4:00pm:
    
    add number of filters into the 1st convolutional layer
    from 64 to 128:
        
        better
        
4:30pm:
    
    add another convolutional layer:
        
        better?

4:50pm:
    
    5 convolutional layers so far, slightly increase
    std of the prediction is 10% for the imaginary part, 2% for real and 1% for radius
    considering remove the max pooling layer in between
    
    worse

"""
