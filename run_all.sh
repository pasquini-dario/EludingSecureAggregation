conf="50 8 0 0"
run="python multiGPU_runs.py"

$run settings.c10_c100 $conf
$run settings.c100_c10 $conf
echo "2/6"

$run settings.c10_tiny $conf
$run settings.c100_tiny $conf
echo "4/6"

$run settings.tiny_c100 $conf
$run settings.tiny_c10 $conf
echo "6/6"