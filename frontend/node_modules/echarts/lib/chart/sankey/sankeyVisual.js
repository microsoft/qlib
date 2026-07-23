
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
import * as zrUtil from 'zrender/lib/core/util.js';
import VisualMapping from '../../visual/VisualMapping.js';
export default function sankeyVisual(ecModel) {
  ecModel.eachSeriesByType('sankey', function (seriesModel) {
    var graph = seriesModel.getGraph();
    var nodes = graph.nodes;
    var edges = graph.edges;
    if (nodes.length) {
      var minValue_1 = Infinity;
      var maxValue_1 = -Infinity;
      zrUtil.each(nodes, function (node) {
        var nodeValue = node.getLayout().value;
        if (nodeValue < minValue_1) {
          minValue_1 = nodeValue;
        }
        if (nodeValue > maxValue_1) {
          maxValue_1 = nodeValue;
        }
      });
      zrUtil.each(nodes, function (node) {
        var mapping = new VisualMapping({
          type: 'color',
          mappingMethod: 'linear',
          dataExtent: [minValue_1, maxValue_1],
          visual: seriesModel.get('color')
        });
        var mapValueToColor = mapping.mapValueToVisual(node.getLayout().value);
        var customColor = node.getModel().get(['itemStyle', 'color']);
        if (customColor != null) {
          node.setVisual('color', customColor);
          node.setVisual('style', {
            fill: customColor
          });
        } else {
          node.setVisual('color', mapValueToColor);
          node.setVisual('style', {
            fill: mapValueToColor
          });
        }
      });
    }
    if (edges.length) {
      zrUtil.each(edges, function (edge) {
        var edgeStyle = edge.getModel().get('lineStyle');
        edge.setVisual('style', edgeStyle);
      });
    }
  });
}