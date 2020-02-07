#!/bin/sh
echo 'hello world'

#start=`date +"%s"`
for (( i=0; i<5; i++ ))
do
    {
        echo ${i}
        # linear
        # nohup python IMBandit.py -imgucb -imfb -seed 300 -p linear -repeat 4 -resdir './SimulationResults/linear_1440' -dataset Flickr > ${i}.txt &
        # nonlinear
        nohup python IMBandit.py -imgucb -imfb -imlinucb -egreedy -ucb1 -seed 300 -prob linear  -repeat 4 -resdir './SimulationResults/linear_1613' -dataset NetHEPT > ${i}.txt &
        # nohup python IMBandit.py -imlinucb -egreedy -ucb1 -seed 300 -p nonlinear  -repeat 1 -resdir './SimulationResults/nonlinear_1557' > ${i}.txt &
    }
    sleep 2
done
# wait
#end=`date +"%s"`
#echo "time: " `expr $end - $start`


# for (( i=20; i<40; i++ ))
# do
#     {
#         echo ${i}
#         nohup python IMBandit.py -imlinucb -egreedy -ucb1 -seed 300 -p linear  -repeat 1 -resdir './SimulationResults/linear_1440' > ${i}.txt &
#     }
#     sleep 1
# done