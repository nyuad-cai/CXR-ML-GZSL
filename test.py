import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer
from arguments import  parse_args


def main ():

    args = parse_args()
    
    try:  
        os.mkdir(args.save_dir)  
    except OSError as error:
        print(error) 
    
    trainer = ChexnetTrainer(args)
    print ('Testing the trained model')
    

    test_ind_auroc = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)
    


    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')
            

if __name__ == '__main__':
    main()





