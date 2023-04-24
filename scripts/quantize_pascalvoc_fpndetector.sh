#!/bin/sh
python quantize.py trained/ai87-pascalvoc-fpndetector-qat8.pth.tar trained/ai87-pascalvoc-fpndetector-qat8-q.pth.tar --device MAX78002 -v
