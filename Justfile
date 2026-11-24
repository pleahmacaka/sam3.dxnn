model := env_var_or_default("SAM3_MODEL", "facebook/sam3")
size := env_var_or_default("SAM3_SIZE", "672")
cuts := env_var_or_default("SAM3_CUTS", "8,16,24")
attention := env_var_or_default("SAM3_ATTENTION", "stock")
image := "sam3-dxnn:cli"
out := "build"

default: build

build: export compile

export:
    uv run sam3-dxnn export --model {{model}} --size {{size}} --cuts {{cuts}} \
        --attention {{attention}} --out {{out}}

compile: _image
    #!/usr/bin/env sh
    set -e
    for config in {{out}}/segment*.json; do
        stem="${config%.json}"
        MSYS_NO_PATHCONV=1 podman run --rm --shm-size=16g \
            -v "$PWD:/work" -e PYTHONPATH=/work/dxcom \
            -e SAM3_CALIB=/work/${stem}_calib.npy {{image}} \
            dxcom -m /work/${stem}_ready.onnx \
                  -c /work/${config} \
                  -o /work/${stem} \
                  --opt_level 0 --aggressive_partitioning --quant_diagnosis
    done

bench picture="calib/dog.jpg": _image
    MSYS_NO_PATHCONV=1 podman run --rm --shm-size=16g -v "$PWD:/work" -e PYTHONPATH=/work/src {{image}} \
        python -m sam3_dxnn bench --build /work/{{out}} --image /work/{{picture}} --size {{size}} --save

compare picture="calib/dog.jpg" prompt="dog": bench
    uv run sam3-dxnn compare --model {{model}} --build {{out}} --image {{picture}} \
        --prompt {{prompt}} --size {{size}}

verify:
    uv run sam3-dxnn verify --model {{model}} --build {{out}} --size {{size}}

_image:
    podman image exists {{image}} || podman build -t {{image}} -f dxcom/Containerfile dxcom

clean:
    rm -rf {{out}}
