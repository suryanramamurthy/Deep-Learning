from math import log
from math import floor

class Zipfian(object):
    """description of class"""

    def __init__(self, skew, dimension):
       self.skew = skew
       self.N = dimension

       if self.skew != 1:
           self.den = (pow(self.N, 1 - self.skew) - 1) / (1 - self.skew) + 0.5 + \
                pow(self.N, -self.skew) / 2 + self.skew / 12 - pow(self.N, -1 - self.skew) * self.skew / 12;
           self.D = 12 * (pow(self.N, -self.skew + 1) - 1) / (1 - self.skew) + 6 + \
                6 * pow(self.N, -self.skew) + self.skew - self.skew * pow(self.N, -self.skew - 1);
       else:
           self.den = log(self.N) + 0.5 + (0.5 / self.N) + (1.0 / 12.0) - (pow(self.N, -2) / 12.0);
           self.D = 12 * log(self.N) + 6 + (6.0 / self.N) + 1 - pow(self.N, -2);

    def zipf_cdf(self,k):
        if k > self.N or k < 1:
            raise Exception("K must be between 1 and N")
        if self.skew != 1:
            num = (pow(k, 1 - self.skew) - 1) / (1 - self.skew) + 0.5 + pow(k, -self.skew) * 0.5 + \
                self.skew / 12 - pow(k, -1 - self.skew) * self.skew / 12;
        else:
            num = log(k) + 0.5 + (0.5 / k) + (1.0 / 12.0) - pow(k, -2) / 12;
        return num / this.den;

    def zip_invcdf(self,p):
        if p > 1 or p < 0:
            raise Exception("probability p must be between 0 and 1");
        tol = 0.01; # if rate of change is below tolerance then stop
        x = self.N / 2.0; # starting value of x (x0)
        pD = p * self.D;
        while True:
            m = pow(x, -self.skew - 2);   # x ^ ( -s - 2) for all values of s
            mx = m * x; # x ^ ( -s - 1) for all values of s
            mxx = mx * x; # x ^ ( -s) for all values of s 
            mxxx = mxx * x; # x ^ ( -s + 1), will not be used when s = 1
            if self.skew != 1:
                num = 12 * (mxxx - 1) / (1 - self.skew) + 6 + 6 * mxx + self.skew - \
                    (self.skew * mx) - pD
                den = 12 * mxx - (6 * self.skew * mx) + (m * self.skew * (self.skew + 1))
            else:
                num = 12 * log(x) + 6 + 6.0 / x + 1 - pow(x, -2) - pD;
                den = 12 / x - 6 * pow(x, -2) + 2 * pow(x, -3);
            nextX = max(1, x - num / den);
            if abs(nextX - x) <= tol:
                return round(nextX);
            x = nextX;
          
    
    



