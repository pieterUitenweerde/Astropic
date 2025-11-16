# astro-stacker
A simple astrophotography image stacking tool written in python.


## TODO:

    - Blob detect KeyError is catastrophic

    - Blob detection algorithm could be improved to  work with all odd shapes. Some u shapes cause missidentifications.
    - Introduce 12, 14, or 32 bit processing
    - Implement weave extract.
    - Blob detect homogonize loop sometimes raises a KeyError when accessing star_table.