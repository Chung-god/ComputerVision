'''
Matching 파트 메인 스크립트
'''

import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/library', './data/library2', ratio_thres=0.6)
    plt.title('Match')
    #plt.imshow(im)

    # Test run matching with ransac
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
    plt.title('MatchRANSAC')
    plt.imshow(im)

if __name__ == '__main__':
    main()
