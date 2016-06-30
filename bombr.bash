for i in {1..1000}
do
  printf $i
  python2 bombr.py -O ai_withflag_new.obser -W ai_withflag_new.reward
done
