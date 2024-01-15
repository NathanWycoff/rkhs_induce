
start=`date +%s`
source .venv/bin/activate

python python/mis.py

#nice -n 10 parallel --jobs 1 --colsep ' ' --will-cite -a sim_args.txt python python/tit_bakeoff.py
nice -n 10 parallel --jobs 1 --colsep ' ' --will-cite -a sim_args.txt python python/hen_bakeoff.py

python python/generic_plot.py

end=`date +%s`
echo $((end-start))
