
#python train_vsl.py -t='vgl' -e firefighters -a='pc' -qp -n=10 -df=1.0 -dfp=1.0 -ename='ADEF_fvgl_' -tn=100 -hz=50 
#python train_vsl.py -t 'vgl' -e roadworld -a='pc' -qp -n=10 -df=1.0 -dfp=1.0 -ename='ADEF_rvgl_' -tn=100 -hz=50 
#python train_vsl.py -t 'vsi' -e roadworld -a 'me' -qp -n=10 -df=1.0 -dfp=1.0 -ename='ADEF_rvsi_' -tn=100 -hz=50 
#python train_vsl.py -t 'vsi' -e firefighters -a 'me' -qp -n=10 -df=1.0 -dfp=1.0 -ename='ADEF_fvsi_' -tn=100 -hz=50
python train_vsl.py -t 'all' -e firefighters -a 'me' -qp -n=10 -df=1.0 -dfp=1.0 -ename='AEPS_fvsl_' -tn=100 -hz=50
python train_vsl.py -t 'all' -e roadworld -a 'me' -qp -n=10 -df=1.0 -dfp=1.0 -ename='AEPS_rvsl_' -tn=100 -hz=50 

