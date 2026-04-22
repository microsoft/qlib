
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
import { eachAfter, eachBefore } from './traversalHelper.js';
import { init, firstWalk, secondWalk, separation as sep, radialCoordinate, getViewRect } from './layoutHelper.js';
export default function treeLayout(ecModel, api) {
  ecModel.eachSeriesByType('tree', function (seriesModel) {
    commonLayout(seriesModel, api);
  });
}
function commonLayout(seriesModel, api) {
  var layoutInfo = getViewRect(seriesModel, api);
  seriesModel.layoutInfo = layoutInfo;
  var layout = seriesModel.get('layout');
  var width = 0;
  var height = 0;
  var separation = null;
  if (layout === 'radial') {
    width = 2 * Math.PI;
    height = Math.min(layoutInfo.height, layoutInfo.width) / 2;
    separation = sep(function (node1, node2) {
      return (node1.parentNode === node2.parentNode ? 1 : 2) / node1.depth;
    });
  } else {
    width = layoutInfo.width;
    height = layoutInfo.height;
    separation = sep();
  }
  var virtualRoot = seriesModel.getData().tree.root;
  var realRoot = virtualRoot.children[0];
  if (realRoot) {
    init(virtualRoot);
    eachAfter(realRoot, firstWalk, separation);
    virtualRoot.hierNode.modifier = -realRoot.hierNode.prelim;
    eachBefore(realRoot, secondWalk);
    var left_1 = realRoot;
    var right_1 = realRoot;
    var bottom_1 = realRoot;
    eachBefore(realRoot, function (node) {
      var x = node.getLayout().x;
      if (x < left_1.getLayout().x) {
        left_1 = node;
      }
      if (x > right_1.getLayout().x) {
        right_1 = node;
      }
      if (node.depth > bottom_1.depth) {
        bottom_1 = node;
      }
    });
    var delta = left_1 === right_1 ? 1 : separation(left_1, right_1) / 2;
    var tx_1 = delta - left_1.getLayout().x;
    var kx_1 = 0;
    var ky_1 = 0;
    var coorX_1 = 0;
    var coorY_1 = 0;
    if (layout === 'radial') {
      kx_1 = width / (right_1.getLayout().x + delta + tx_1);
      // here we use (node.depth - 1), bucause the real root's depth is 1
      ky_1 = height / (bottom_1.depth - 1 || 1);
      eachBefore(realRoot, function (node) {
        coorX_1 = (node.getLayout().x + tx_1) * kx_1;
        coorY_1 = (node.depth - 1) * ky_1;
        var finalCoor = radialCoordinate(coorX_1, coorY_1);
        node.setLayout({
          x: finalCoor.x,
          y: finalCoor.y,
          rawX: coorX_1,
          rawY: coorY_1
        }, true);
      });
    } else {
      var orient_1 = seriesModel.getOrient();
      if (orient_1 === 'RL' || orient_1 === 'LR') {
        ky_1 = height / (right_1.getLayout().x + delta + tx_1);
        kx_1 = width / (bottom_1.depth - 1 || 1);
        eachBefore(realRoot, function (node) {
          coorY_1 = (node.getLayout().x + tx_1) * ky_1;
          coorX_1 = orient_1 === 'LR' ? (node.depth - 1) * kx_1 : width - (node.depth - 1) * kx_1;
          node.setLayout({
            x: coorX_1,
            y: coorY_1
          }, true);
        });
      } else if (orient_1 === 'TB' || orient_1 === 'BT') {
        kx_1 = width / (right_1.getLayout().x + delta + tx_1);
        ky_1 = height / (bottom_1.depth - 1 || 1);
        eachBefore(realRoot, function (node) {
          coorX_1 = (node.getLayout().x + tx_1) * kx_1;
          coorY_1 = orient_1 === 'TB' ? (node.depth - 1) * ky_1 : height - (node.depth - 1) * ky_1;
          node.setLayout({
            x: coorX_1,
            y: coorY_1
          }, true);
        });
      }
    }
  }
}