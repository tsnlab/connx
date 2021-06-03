FILES=`find gen -type f`

cat << EOF > main/CMakeLists.txt
idf_component_register(SRCS "main.c"
EOF

for FILE in $FILES
do
cat << EOF >> main/CMakeLists.txt
                       "../${FILE}"
EOF
done

cat << EOF >> main/CMakeLists.txt
                       INCLUDE_DIRS "../../../include")

spiffs_create_partition_image(storage ../spiffs FLASH_IN_PROJECT)
EOF
