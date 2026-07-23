
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
/*
* A third-party license is embedded for some of the code in this file:
* Some formulas were originally copied from "d3.js" with some
* modifications made for this project.
* (See more details in the comment of the method "step" below.)
* The use of the source code of this file is also subject to the terms
* and consitions of the license of "d3.js" (BSD-3Clause, see
* </licenses/LICENSE-d3>).
*/
import * as vec2 from 'zrender/lib/core/vector.js';
var scaleAndAdd = vec2.scaleAndAdd;
// function adjacentNode(n, e) {
//     return e.n1 === n ? e.n2 : e.n1;
// }
export function forceLayout(inNodes, inEdges, opts) {
  var nodes = inNodes;
  var edges = inEdges;
  var rect = opts.rect;
  var width = rect.width;
  var height = rect.height;
  var center = [rect.x + width / 2, rect.y + height / 2];
  // let scale = opts.scale || 1;
  var gravity = opts.gravity == null ? 0.1 : opts.gravity;
  // for (let i = 0; i < edges.length; i++) {
  //     let e = edges[i];
  //     let n1 = e.n1;
  //     let n2 = e.n2;
  //     n1.edges = n1.edges || [];
  //     n2.edges = n2.edges || [];
  //     n1.edges.push(e);
  //     n2.edges.push(e);
  // }
  // Init position
  for (var i = 0; i < nodes.length; i++) {
    var n = nodes[i];
    if (!n.p) {
      n.p = vec2.create(width * (Math.random() - 0.5) + center[0], height * (Math.random() - 0.5) + center[1]);
    }
    n.pp = vec2.clone(n.p);
    n.edges = null;
  }
  // Formula in 'Graph Drawing by Force-directed Placement'
  // let k = scale * Math.sqrt(width * height / nodes.length);
  // let k2 = k * k;
  var initialFriction = opts.friction == null ? 0.6 : opts.friction;
  var friction = initialFriction;
  var beforeStepCallback;
  var afterStepCallback;
  return {
    warmUp: function () {
      friction = initialFriction * 0.8;
    },
    setFixed: function (idx) {
      nodes[idx].fixed = true;
    },
    setUnfixed: function (idx) {
      nodes[idx].fixed = false;
    },
    /**
     * Before step hook
     */
    beforeStep: function (cb) {
      beforeStepCallback = cb;
    },
    /**
     * After step hook
     */
    afterStep: function (cb) {
      afterStepCallback = cb;
    },
    /**
     * Some formulas were originally copied from "d3.js"
     * https://github.com/d3/d3/blob/b516d77fb8566b576088e73410437494717ada26/src/layout/force.js
     * with some modifications made for this project.
     * See the license statement at the head of this file.
     */
    step: function (cb) {
      beforeStepCallback && beforeStepCallback(nodes, edges);
      var v12 = [];
      var nLen = nodes.length;
      for (var i = 0; i < edges.length; i++) {
        var e = edges[i];
        if (e.ignoreForceLayout) {
          continue;
        }
        var n1 = e.n1;
        var n2 = e.n2;
        vec2.sub(v12, n2.p, n1.p);
        var d = vec2.len(v12) - e.d;
        var w = n2.w / (n1.w + n2.w);
        if (isNaN(w)) {
          w = 0;
        }
        vec2.normalize(v12, v12);
        !n1.fixed && scaleAndAdd(n1.p, n1.p, v12, w * d * friction);
        !n2.fixed && scaleAndAdd(n2.p, n2.p, v12, -(1 - w) * d * friction);
      }
      // Gravity
      for (var i = 0; i < nLen; i++) {
        var n = nodes[i];
        if (!n.fixed) {
          vec2.sub(v12, center, n.p);
          // let d = vec2.len(v12);
          // vec2.scale(v12, v12, 1 / d);
          // let gravityFactor = gravity;
          scaleAndAdd(n.p, n.p, v12, gravity * friction);
        }
      }
      // Repulsive
      // PENDING
      for (var i = 0; i < nLen; i++) {
        var n1 = nodes[i];
        for (var j = i + 1; j < nLen; j++) {
          var n2 = nodes[j];
          vec2.sub(v12, n2.p, n1.p);
          var d = vec2.len(v12);
          if (d === 0) {
            // Random repulse
            vec2.set(v12, Math.random() - 0.5, Math.random() - 0.5);
            d = 1;
          }
          var repFact = (n1.rep + n2.rep) / d / d;
          !n1.fixed && scaleAndAdd(n1.pp, n1.pp, v12, repFact);
          !n2.fixed && scaleAndAdd(n2.pp, n2.pp, v12, -repFact);
        }
      }
      var v = [];
      for (var i = 0; i < nLen; i++) {
        var n = nodes[i];
        if (!n.fixed) {
          vec2.sub(v, n.p, n.pp);
          scaleAndAdd(n.p, n.p, v, friction);
          vec2.copy(n.pp, n.p);
        }
      }
      friction = friction * 0.992;
      var finished = friction < 0.01;
      afterStepCallback && afterStepCallback(nodes, edges, finished);
      cb && cb(finished);
    }
  };
}