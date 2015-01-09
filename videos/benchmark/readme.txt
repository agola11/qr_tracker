Explanation of results:

We did two separate trials, each time running the full algorithm with/without
hsv color filtering, IIR corner filtering, and cropping. The results are
named accordingly. Adjusted time is computed by dividing raw computation time by
accuracy, which is the percent of the video frames in which the algorithm detects
a valid bounding box.

ALGORITHM    RUN    COMPUTATION_TIME  ADJUSTED_TIME   VALID_TIME  TOTAL_TIME   Accuracy
full	     1	    14.47	      37.1	      18.6	  48	       0.39
full	     2	    13.63	      23.9	      28.0	  49	       0.57
no IIR	     1	    13.30	      42.9	      13.1  	  42	       0.31
no IIR	     2	    15.55	      111.1	      6.2	  44	       0.14
no hsv	     1	    9.8		      25.8	      16.1	  42	       0.38
no hsv	     2	    11.20	      44.8	      12.6	  50	       0.25
no crop	     1	    14.74	      35.1	      15.6 	  37	       0.42
no crop	     2	    17.43	      39.6	      19.2	  44	       0.44
