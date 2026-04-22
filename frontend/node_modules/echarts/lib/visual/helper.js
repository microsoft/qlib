
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/


/**
 * AUTO-GENERATED FILE. DO NOT MODIFY.
 */

/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
export function getItemVisualFromData(data, dataIndex, key) {
  switch (key) {
    case 'color':
      var style = data.getItemVisual(dataIndex, 'style');
      return style[data.getVisual('drawType')];
    case 'opacity':
      return data.getItemVisual(dataIndex, 'style').opacity;
    case 'symbol':
    case 'symbolSize':
    case 'liftZ':
      return data.getItemVisual(dataIndex, key);
    default:
      if (process.env.NODE_ENV !== 'production') {
        console.warn("Unknown visual type " + key);
      }
  }
}
export function getVisualFromData(data, key) {
  switch (key) {
    case 'color':
      var style = data.getVisual('style');
      return style[data.getVisual('drawType')];
    case 'opacity':
      return data.getVisual('style').opacity;
    case 'symbol':
    case 'symbolSize':
    case 'liftZ':
      return data.getVisual(key);
    default:
      if (process.env.NODE_ENV !== 'production') {
        console.warn("Unknown visual type " + key);
      }
  }
}
export function setItemVisualFromData(data, dataIndex, key, value) {
  switch (key) {
    case 'color':
      // Make sure not sharing style object.
      var style = data.ensureUniqueItemVisual(dataIndex, 'style');
      style[data.getVisual('drawType')] = value;
      // Mark the color has been changed, not from palette anymore
      data.setItemVisual(dataIndex, 'colorFromPalette', false);
      break;
    case 'opacity':
      data.ensureUniqueItemVisual(dataIndex, 'style').opacity = value;
      break;
    case 'symbol':
    case 'symbolSize':
    case 'liftZ':
      data.setItemVisual(dataIndex, key, value);
      break;
    default:
      if (process.env.NODE_ENV !== 'production') {
        console.warn("Unknown visual type " + key);
      }
  }
}