
for filename in `ls *.log`
do
    cat ${filename} | tee >(awk '!/\r/' >> ${filename}.txt )
done
