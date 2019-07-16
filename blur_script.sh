for i in `seq 1 20`
do
    if [ $i -ne 12 ] 
    then
        echo /home/ubuntu/gsarar/dataset/case${i}/P${i}_dcm
        python task2.py --i /home/ubuntu/gsarar/dataset/case${i}/P${i}_dcm --o /home/ubuntu/gsarar/dataset/case${i}_blurred
    fi
done

for i in `seq 1 20`
do
    if [ $i -ne 12 ] 
    then
        echo $i
        python task1_part1.py --i /home/ubuntu/gsarar/dataset/case${i}_blurred --h datasetHFD5/blurred${i}.h5
    fi
done

for i in `seq 1 20`
do
    if [ $i -ne 12 ] 
    then
        echo $i
        python task1_part1.py --i /home/ubuntu/gsarar/dataset/case${i}/P${i}_dcm --h datasetHFD5/clean${i}.h5
    fi
done