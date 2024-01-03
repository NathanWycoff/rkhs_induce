
start=`date +%s`
source .venv/bin/activate

python python/mis.py

nice -n 10 parallel --jobs 30 --colsep ' ' --will-cite -a sim_args.txt python python/tit_bakeoff.py

python python/tit_plot.py

end=`date +%s`
echo $((end-start))
