-----------------------------------------
| Author: Serif Seremet					|
| Date: 21.07.2021						|
| Software Architecture Research Group	|
-----------------------------------------

Implementation of the Bat4CEP project. 

Further information about the algorithm or the structure of the datasets can be found in the corresponding paper 
"Bat4CEP: A Bat Algorithm for Mining of Complex Event Processing Rules" by R. Bruns and J. Dunkel. 

************** GETTING STARTED **************

The main method of this program can be found in BatTestMain.java, where all test cases are defined. If you start the project without any changes, 
the first test case presented in the Bat4CEP paper will be executed. 

All other test cases described in the paper are included as comments in the main method of BatTestMain.java. If you want to reproduce these test cases, 
it is sufficient to uncomment them. Since Bat4CEP is a algorithm with a lot of random behavior, the results cannot be replicated exactly. 
Small deviations are therefore to be expected.

For importing, make sure to import the implementation as a Maven project.
Perform a maven-update if the development environment shows errors.

************** NOTE **************

If you want to test the implementation on a small environment with few resources, it is recommended to reduce 
the swarm size in BatTestMain.java from 500 to e.g. 200 and the number of threads from 32 to e.g. 5. 
Even with a smaller swarm size, Bat4CEP delivers good results.

However, the results of the paper can only be guaranteed with the default configurations. These were tested on a cloud infrastructure with 32 processors and 64 GB RAM.
