FILES=`find gen -name *.c`

TMP=`mktemp`

cat << EOF > $TMP
idf_component_register(SRCS "main.c" "cam.c"
EOF

for FILE in $FILES
do
cat << EOF >> $TMP
                       "../${FILE}"
EOF
done

cat << EOF >> $TMP
                       INCLUDE_DIRS "../../../include" "include"
                       REQUIRES esp32-camera spiffs)

spiffs_create_partition_image(storage ../spiffs FLASH_IN_PROJECT)
EOF

DIFF=`diff $TMP main/CMakeLists.txt | wc -l`
if [ $DIFF != "0" ]; then
	cp $TMP main/CMakeLists.txt
fi

rm $TMP
