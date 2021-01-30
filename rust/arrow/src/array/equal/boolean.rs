// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::array::ArrayData;

use super::utils::equal_bits;

pub(super) fn boolean_equal(
    lhs: &ArrayData,
    rhs: &ArrayData,
    lhs_start: usize,
    rhs_start: usize,
    len: usize,
) -> bool {
    let lhs_values = lhs.buffers()[0].as_slice();
    let rhs_values = rhs.buffers()[0].as_slice();

    // TODO: we can do this more efficiently if all values are not-null
    (0..len).all(|i| {
        let lhs_pos = lhs_start + i;
        let rhs_pos = rhs_start + i;
        let lhs_is_null = lhs.is_null(lhs_pos);
        let rhs_is_null = rhs.is_null(rhs_pos);

        lhs_is_null
            || (lhs_is_null == rhs_is_null)
                && equal_bits(
                    lhs_values,
                    rhs_values,
                    lhs_pos + lhs.offset(),
                    rhs_pos + rhs.offset(),
                    1,
                )
    })
}
