
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
import { lift } from 'zrender/lib/tool/color.js';
import { extend, isString } from 'zrender/lib/core/util.js';
export default function sunburstVisual(ecModel) {
  var paletteScope = {};
  // Default color strategy
  function pickColor(node, seriesModel, treeHeight) {
    // Choose color from palette based on the first level.
    var current = node;
    while (current && current.depth > 1) {
      current = current.parentNode;
    }
    var color = seriesModel.getColorFromPalette(current.name || current.dataIndex + '', paletteScope);
    if (node.depth > 1 && isString(color)) {
      // Lighter on the deeper level.
      color = lift(color, (node.depth - 1) / (treeHeight - 1) * 0.5);
    }
    return color;
  }
  ecModel.eachSeriesByType('sunburst', function (seriesModel) {
    var data = seriesModel.getData();
    var tree = data.tree;
    tree.eachNode(function (node) {
      var model = node.getModel();
      var style = model.getModel('itemStyle').getItemStyle();
      if (!style.fill) {
        style.fill = pickColor(node, seriesModel, tree.root.height);
      }
      var existsStyle = data.ensureUniqueItemVisual(node.dataIndex, 'style');
      extend(existsStyle, style);
    });
  });
}