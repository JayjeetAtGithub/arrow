// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package utils_test

import (
	"reflect"
	"testing"

	"github.com/apache/arrow/go/arrow/bitutil"
	"github.com/apache/arrow/go/parquet/internal/utils"
	"github.com/stretchr/testify/suite"
)

func reverseAny(s interface{}) {
	n := reflect.ValueOf(s).Len()
	swap := reflect.Swapper(s)
	for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
		swap(i, j)
	}
}

type linearBitRunReader struct {
	reader *utils.BitmapReader
}

func (l linearBitRunReader) NextRun() utils.BitRun {
	r := utils.BitRun{0, l.reader.Set()}
	for l.reader.Pos() < l.reader.Len() && l.reader.Set() == r.Set {
		r.Len++
		l.reader.Next()
	}
	return r
}

func bitmapFromString(s string) []byte {
	maxLen := bitutil.BytesForBits(int64(len(s)))
	ret := make([]byte, maxLen)
	i := 0
	for _, c := range s {
		switch c {
		case '0':
			bitutil.ClearBit(ret, i)
			i++
		case '1':
			bitutil.SetBit(ret, i)
			i++
		case ' ', '\t', '\r', '\n':
		default:
			panic("unexpected character for bitmap string")
		}
	}

	actualLen := bitutil.BytesForBits(int64(i))
	return ret[:actualLen]
}

func referenceBitRuns(data []byte, offset, length int64) (ret []utils.SetBitRun) {
	ret = make([]utils.SetBitRun, 0)
	reader := linearBitRunReader{utils.NewBitmapReader(data, offset, length)}
	pos := int64(0)
	for pos < length {
		br := reader.NextRun()
		if br.Set {
			ret = append(ret, utils.SetBitRun{pos, br.Len})
		}
		pos += br.Len
	}
	return
}

type BitSetRunReaderSuite struct {
	suite.Suite

	testOffsets []int64
}

func TestBitSetRunReader(t *testing.T) {
	suite.Run(t, new(BitSetRunReaderSuite))
}

func (br *BitSetRunReaderSuite) SetupSuite() {
	br.testOffsets = []int64{0, 1, 6, 7, 8, 33, 63, 64, 65, 71}
}

func (br *BitSetRunReaderSuite) SetupTest() {
	br.T().Parallel()
}

type Range struct {
	Offset int64
	Len    int64
}

func (r Range) EndOffset() int64 { return r.Offset + r.Len }

func (br *BitSetRunReaderSuite) bufferTestRanges(buf []byte) []Range {
	bufSize := int64(len(buf) * 8) // in bits
	rg := make([]Range, 0)
	for _, offset := range br.testOffsets {
		for _, lenAdjust := range br.testOffsets {
			length := utils.Min(bufSize-offset, lenAdjust)
			br.GreaterOrEqual(length, int64(0))
			rg = append(rg, Range{offset, length})
			length = utils.Min(bufSize-offset, bufSize-lenAdjust)
			br.GreaterOrEqual(length, int64(0))
			rg = append(rg, Range{offset, length})
		}
	}
	return rg
}

func (br *BitSetRunReaderSuite) assertBitRuns(buf []byte, start, length int64, expected []utils.SetBitRun) {
	{
		runs := make([]utils.SetBitRun, 0)
		reader := utils.NewSetBitRunReader(buf, start, length)
		for {
			run := reader.NextRun()
			if run.Length == 0 {
				break
			}
			runs = append(runs, run)
		}
		br.Equal(expected, runs)
	}
	{
		runs := make([]utils.SetBitRun, 0)
		reader := utils.NewReverseSetBitRunReader(buf, start, length)
		for {
			run := reader.NextRun()
			if run.Length == 0 {
				break
			}
			runs = append(runs, run)
		}
		reverseAny(expected)
		br.Equal(expected, runs)
	}
}

func (br *BitSetRunReaderSuite) TestEmpty() {
	for _, offset := range br.testOffsets {
		br.assertBitRuns(nil, offset, 0, []utils.SetBitRun{})
	}
}

func (br *BitSetRunReaderSuite) TestOneByte() {
	buffer := bitmapFromString("01101101")
	br.assertBitRuns(buffer, 0, 8, []utils.SetBitRun{
		{1, 2}, {4, 2}, {7, 1},
	})

	for _, str := range []string{"01101101", "10110110", "00000000", "11111111"} {
		buf := bitmapFromString(str)
		for offset := int64(0); offset < 8; offset++ {
			for length := int64(0); length <= 8-offset; length++ {
				expected := referenceBitRuns(buf, offset, length)
				br.assertBitRuns(buf, offset, length, expected)
			}
		}
	}
}

func (br *BitSetRunReaderSuite) TestTiny() {
	buf := bitmapFromString("11100011 10001110 00111000 11100011 10001110 00111000")

	br.assertBitRuns(buf, 0, 48, []utils.SetBitRun{
		{0, 3}, {6, 3}, {12, 3}, {18, 3}, {24, 3}, {30, 3}, {36, 3}, {42, 3},
	})
	br.assertBitRuns(buf, 0, 46, []utils.SetBitRun{
		{0, 3}, {6, 3}, {12, 3}, {18, 3}, {24, 3}, {30, 3}, {36, 3}, {42, 3},
	})
	br.assertBitRuns(buf, 0, 45, []utils.SetBitRun{
		{0, 3}, {6, 3}, {12, 3}, {18, 3}, {24, 3}, {30, 3}, {36, 3}, {42, 3},
	})
	br.assertBitRuns(buf, 0, 42, []utils.SetBitRun{
		{0, 3}, {6, 3}, {12, 3}, {18, 3}, {24, 3}, {30, 3}, {36, 3},
	})
	br.assertBitRuns(buf, 3, 45, []utils.SetBitRun{
		{3, 3}, {9, 3}, {15, 3}, {21, 3}, {27, 3}, {33, 3}, {39, 3},
	})
	br.assertBitRuns(buf, 3, 43, []utils.SetBitRun{
		{3, 3}, {9, 3}, {15, 3}, {21, 3}, {27, 3}, {33, 3}, {39, 3},
	})
	br.assertBitRuns(buf, 3, 42, []utils.SetBitRun{
		{3, 3}, {9, 3}, {15, 3}, {21, 3}, {27, 3}, {33, 3}, {39, 3},
	})
	br.assertBitRuns(buf, 3, 39, []utils.SetBitRun{
		{3, 3}, {9, 3}, {15, 3}, {21, 3}, {27, 3}, {33, 3},
	})
}

func (br *BitSetRunReaderSuite) TestAllZeros() {
	const bufferSize = 256
	buf := make([]byte, int(bitutil.BytesForBits(bufferSize)))

	for _, rg := range br.bufferTestRanges(buf) {
		br.assertBitRuns(buf, rg.Offset, rg.Len, []utils.SetBitRun{})
	}
}

func (br *BitSetRunReaderSuite) TestAllOnes() {
	const bufferSize = 256
	buf := make([]byte, int(bitutil.BytesForBits(bufferSize)))
	utils.SetBitsTo(buf, 0, bufferSize, true)

	for _, rg := range br.bufferTestRanges(buf) {
		if rg.Len > 0 {
			br.assertBitRuns(buf, rg.Offset, rg.Len, []utils.SetBitRun{{0, rg.Len}})
		} else {
			br.assertBitRuns(buf, rg.Offset, rg.Len, []utils.SetBitRun{})
		}
	}
}

func (br *BitSetRunReaderSuite) TestSmall() {
	// ones then zeros then ones
	const (
		bufferSize      = 256
		onesLen         = 64
		secondOnesStart = bufferSize - onesLen
	)

	buf := make([]byte, int(bitutil.BytesForBits(bufferSize)))
	utils.SetBitsTo(buf, 0, bufferSize, false)
	utils.SetBitsTo(buf, 0, onesLen, true)
	utils.SetBitsTo(buf, secondOnesStart, onesLen, true)

	for _, rg := range br.bufferTestRanges(buf) {
		expected := []utils.SetBitRun{}
		if rg.Offset < onesLen && rg.Len > 0 {
			expected = append(expected, utils.SetBitRun{0, utils.Min(onesLen-rg.Offset, rg.Len)})
		}
		if rg.Offset+rg.Len > secondOnesStart {
			expected = append(expected, utils.SetBitRun{secondOnesStart - rg.Offset, rg.Len + rg.Offset - secondOnesStart})
		}
		br.assertBitRuns(buf, rg.Offset, rg.Len, expected)
	}
}

func (br *BitSetRunReaderSuite) TestSingleRun() {
	// one single run of ones, at varying places in the buffer
	const bufferSize = 512
	buf := make([]byte, int(bitutil.BytesForBits(bufferSize)))

	for _, onesRg := range br.bufferTestRanges(buf) {
		utils.SetBitsTo(buf, 0, bufferSize, false)
		utils.SetBitsTo(buf, onesRg.Offset, onesRg.Len, true)

		for _, rg := range br.bufferTestRanges(buf) {
			expect := []utils.SetBitRun{}
			if rg.Len != 0 && onesRg.Len != 0 && rg.Offset < onesRg.EndOffset() && onesRg.Offset < rg.EndOffset() {
				// the two ranges intersect
				var (
					intersectStart = utils.Max(rg.Offset, onesRg.Offset)
					intersectStop  = utils.Min(rg.EndOffset(), onesRg.EndOffset())
				)
				expect = append(expect, utils.SetBitRun{intersectStart - rg.Offset, intersectStop - intersectStart})
			}
			br.assertBitRuns(buf, rg.Offset, rg.Len, expect)
		}
	}
}
