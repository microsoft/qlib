
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
// id may be function name of Object, add a prefix to avoid this problem.
function generateNodeKey(id) {
  return '_EC_' + id;
}
var Graph = /** @class */function () {
  function Graph(directed) {
    this.type = 'graph';
    this.nodes = [];
    this.edges = [];
    this._nodesMap = {};
    /**
     * @type {Object.<string, module:echarts/data/Graph.Edge>}
     * @private
     */
    this._edgesMap = {};
    this._directed = directed || false;
  }
  /**
   * If is directed graph
   */
  Graph.prototype.isDirected = function () {
    return this._directed;
  };
  ;
  /**
   * Add a new node
   */
  Graph.prototype.addNode = function (id, dataIndex) {
    id = id == null ? '' + dataIndex : '' + id;
    var nodesMap = this._nodesMap;
    if (nodesMap[generateNodeKey(id)]) {
      if (process.env.NODE_ENV !== 'production') {
        console.error('Graph nodes have duplicate name or id');
      }
      return;
    }
    var node = new GraphNode(id, dataIndex);
    node.hostGraph = this;
    this.nodes.push(node);
    nodesMap[generateNodeKey(id)] = node;
    return node;
  };
  ;
  /**
   * Get node by data index
   */
  Graph.prototype.getNodeByIndex = function (dataIndex) {
    var rawIdx = this.data.getRawIndex(dataIndex);
    return this.nodes[rawIdx];
  };
  ;
  /**
   * Get node by id
   */
  Graph.prototype.getNodeById = function (id) {
    return this._nodesMap[generateNodeKey(id)];
  };
  ;
  /**
   * Add a new edge
   */
  Graph.prototype.addEdge = function (n1, n2, dataIndex) {
    var nodesMap = this._nodesMap;
    var edgesMap = this._edgesMap;
    // PENDING
    if (zrUtil.isNumber(n1)) {
      n1 = this.nodes[n1];
    }
    if (zrUtil.isNumber(n2)) {
      n2 = this.nodes[n2];
    }
    if (!(n1 instanceof GraphNode)) {
      n1 = nodesMap[generateNodeKey(n1)];
    }
    if (!(n2 instanceof GraphNode)) {
      n2 = nodesMap[generateNodeKey(n2)];
    }
    if (!n1 || !n2) {
      return;
    }
    var key = n1.id + '-' + n2.id;
    var edge = new GraphEdge(n1, n2, dataIndex);
    edge.hostGraph = this;
    if (this._directed) {
      n1.outEdges.push(edge);
      n2.inEdges.push(edge);
    }
    n1.edges.push(edge);
    if (n1 !== n2) {
      n2.edges.push(edge);
    }
    this.edges.push(edge);
    edgesMap[key] = edge;
    return edge;
  };
  ;
  /**
   * Get edge by data index
   */
  Graph.prototype.getEdgeByIndex = function (dataIndex) {
    var rawIdx = this.edgeData.getRawIndex(dataIndex);
    return this.edges[rawIdx];
  };
  ;
  /**
   * Get edge by two linked nodes
   */
  Graph.prototype.getEdge = function (n1, n2) {
    if (n1 instanceof GraphNode) {
      n1 = n1.id;
    }
    if (n2 instanceof GraphNode) {
      n2 = n2.id;
    }
    var edgesMap = this._edgesMap;
    if (this._directed) {
      return edgesMap[n1 + '-' + n2];
    } else {
      return edgesMap[n1 + '-' + n2] || edgesMap[n2 + '-' + n1];
    }
  };
  ;
  /**
   * Iterate all nodes
   */
  Graph.prototype.eachNode = function (cb, context) {
    var nodes = this.nodes;
    var len = nodes.length;
    for (var i = 0; i < len; i++) {
      if (nodes[i].dataIndex >= 0) {
        cb.call(context, nodes[i], i);
      }
    }
  };
  ;
  /**
   * Iterate all edges
   */
  Graph.prototype.eachEdge = function (cb, context) {
    var edges = this.edges;
    var len = edges.length;
    for (var i = 0; i < len; i++) {
      if (edges[i].dataIndex >= 0 && edges[i].node1.dataIndex >= 0 && edges[i].node2.dataIndex >= 0) {
        cb.call(context, edges[i], i);
      }
    }
  };
  ;
  /**
   * Breadth first traverse
   * Return true to stop traversing
   */
  Graph.prototype.breadthFirstTraverse = function (cb, startNode, direction, context) {
    if (!(startNode instanceof GraphNode)) {
      startNode = this._nodesMap[generateNodeKey(startNode)];
    }
    if (!startNode) {
      return;
    }
    var edgeType = direction === 'out' ? 'outEdges' : direction === 'in' ? 'inEdges' : 'edges';
    for (var i = 0; i < this.nodes.length; i++) {
      this.nodes[i].__visited = false;
    }
    if (cb.call(context, startNode, null)) {
      return;
    }
    var queue = [startNode];
    while (queue.length) {
      var currentNode = queue.shift();
      var edges = currentNode[edgeType];
      for (var i = 0; i < edges.length; i++) {
        var e = edges[i];
        var otherNode = e.node1 === currentNode ? e.node2 : e.node1;
        if (!otherNode.__visited) {
          if (cb.call(context, otherNode, currentNode)) {
            // Stop traversing
            return;
          }
          queue.push(otherNode);
          otherNode.__visited = true;
        }
      }
    }
  };
  ;
  // TODO
  // depthFirstTraverse(
  //     cb, startNode, direction, context
  // ) {
  // };
  // Filter update
  Graph.prototype.update = function () {
    var data = this.data;
    var edgeData = this.edgeData;
    var nodes = this.nodes;
    var edges = this.edges;
    for (var i = 0, len = nodes.length; i < len; i++) {
      nodes[i].dataIndex = -1;
    }
    for (var i = 0, len = data.count(); i < len; i++) {
      nodes[data.getRawIndex(i)].dataIndex = i;
    }
    edgeData.filterSelf(function (idx) {
      var edge = edges[edgeData.getRawIndex(idx)];
      return edge.node1.dataIndex >= 0 && edge.node2.dataIndex >= 0;
    });
    // Update edge
    for (var i = 0, len = edges.length; i < len; i++) {
      edges[i].dataIndex = -1;
    }
    for (var i = 0, len = edgeData.count(); i < len; i++) {
      edges[edgeData.getRawIndex(i)].dataIndex = i;
    }
  };
  ;
  /**
   * @return {module:echarts/data/Graph}
   */
  Graph.prototype.clone = function () {
    var graph = new Graph(this._directed);
    var nodes = this.nodes;
    var edges = this.edges;
    for (var i = 0; i < nodes.length; i++) {
      graph.addNode(nodes[i].id, nodes[i].dataIndex);
    }
    for (var i = 0; i < edges.length; i++) {
      var e = edges[i];
      graph.addEdge(e.node1.id, e.node2.id, e.dataIndex);
    }
    return graph;
  };
  ;
  return Graph;
}();
var GraphNode = /** @class */function () {
  function GraphNode(id, dataIndex) {
    this.inEdges = [];
    this.outEdges = [];
    this.edges = [];
    this.dataIndex = -1;
    this.id = id == null ? '' : id;
    this.dataIndex = dataIndex == null ? -1 : dataIndex;
  }
  /**
   * @return {number}
   */
  GraphNode.prototype.degree = function () {
    return this.edges.length;
  };
  /**
   * @return {number}
   */
  GraphNode.prototype.inDegree = function () {
    return this.inEdges.length;
  };
  /**
  * @return {number}
  */
  GraphNode.prototype.outDegree = function () {
    return this.outEdges.length;
  };
  GraphNode.prototype.getModel = function (path) {
    if (this.dataIndex < 0) {
      return;
    }
    var graph = this.hostGraph;
    var itemModel = graph.data.getItemModel(this.dataIndex);
    return itemModel.getModel(path);
  };
  GraphNode.prototype.getAdjacentDataIndices = function () {
    var dataIndices = {
      edge: [],
      node: []
    };
    for (var i = 0; i < this.edges.length; i++) {
      var adjacentEdge = this.edges[i];
      if (adjacentEdge.dataIndex < 0) {
        continue;
      }
      dataIndices.edge.push(adjacentEdge.dataIndex);
      dataIndices.node.push(adjacentEdge.node1.dataIndex, adjacentEdge.node2.dataIndex);
    }
    return dataIndices;
  };
  GraphNode.prototype.getTrajectoryDataIndices = function () {
    var connectedEdgesMap = zrUtil.createHashMap();
    var connectedNodesMap = zrUtil.createHashMap();
    for (var i = 0; i < this.edges.length; i++) {
      var adjacentEdge = this.edges[i];
      if (adjacentEdge.dataIndex < 0) {
        continue;
      }
      connectedEdgesMap.set(adjacentEdge.dataIndex, true);
      var sourceNodesQueue = [adjacentEdge.node1];
      var targetNodesQueue = [adjacentEdge.node2];
      var nodeIteratorIndex = 0;
      while (nodeIteratorIndex < sourceNodesQueue.length) {
        var sourceNode = sourceNodesQueue[nodeIteratorIndex];
        nodeIteratorIndex++;
        connectedNodesMap.set(sourceNode.dataIndex, true);
        for (var j = 0; j < sourceNode.inEdges.length; j++) {
          connectedEdgesMap.set(sourceNode.inEdges[j].dataIndex, true);
          sourceNodesQueue.push(sourceNode.inEdges[j].node1);
        }
      }
      nodeIteratorIndex = 0;
      while (nodeIteratorIndex < targetNodesQueue.length) {
        var targetNode = targetNodesQueue[nodeIteratorIndex];
        nodeIteratorIndex++;
        connectedNodesMap.set(targetNode.dataIndex, true);
        for (var j = 0; j < targetNode.outEdges.length; j++) {
          connectedEdgesMap.set(targetNode.outEdges[j].dataIndex, true);
          targetNodesQueue.push(targetNode.outEdges[j].node2);
        }
      }
    }
    return {
      edge: connectedEdgesMap.keys(),
      node: connectedNodesMap.keys()
    };
  };
  return GraphNode;
}();
var GraphEdge = /** @class */function () {
  function GraphEdge(n1, n2, dataIndex) {
    this.dataIndex = -1;
    this.node1 = n1;
    this.node2 = n2;
    this.dataIndex = dataIndex == null ? -1 : dataIndex;
  }
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  GraphEdge.prototype.getModel = function (path) {
    if (this.dataIndex < 0) {
      return;
    }
    var graph = this.hostGraph;
    var itemModel = graph.edgeData.getItemModel(this.dataIndex);
    return itemModel.getModel(path);
  };
  GraphEdge.prototype.getAdjacentDataIndices = function () {
    return {
      edge: [this.dataIndex],
      node: [this.node1.dataIndex, this.node2.dataIndex]
    };
  };
  GraphEdge.prototype.getTrajectoryDataIndices = function () {
    var connectedEdgesMap = zrUtil.createHashMap();
    var connectedNodesMap = zrUtil.createHashMap();
    connectedEdgesMap.set(this.dataIndex, true);
    var sourceNodes = [this.node1];
    var targetNodes = [this.node2];
    var nodeIteratorIndex = 0;
    while (nodeIteratorIndex < sourceNodes.length) {
      var sourceNode = sourceNodes[nodeIteratorIndex];
      nodeIteratorIndex++;
      connectedNodesMap.set(sourceNode.dataIndex, true);
      for (var j = 0; j < sourceNode.inEdges.length; j++) {
        connectedEdgesMap.set(sourceNode.inEdges[j].dataIndex, true);
        sourceNodes.push(sourceNode.inEdges[j].node1);
      }
    }
    nodeIteratorIndex = 0;
    while (nodeIteratorIndex < targetNodes.length) {
      var targetNode = targetNodes[nodeIteratorIndex];
      nodeIteratorIndex++;
      connectedNodesMap.set(targetNode.dataIndex, true);
      for (var j = 0; j < targetNode.outEdges.length; j++) {
        connectedEdgesMap.set(targetNode.outEdges[j].dataIndex, true);
        targetNodes.push(targetNode.outEdges[j].node2);
      }
    }
    return {
      edge: connectedEdgesMap.keys(),
      node: connectedNodesMap.keys()
    };
  };
  return GraphEdge;
}();
function createGraphDataProxyMixin(hostName, dataName) {
  return {
    /**
     * @param Default 'value'. can be 'a', 'b', 'c', 'd', 'e'.
     */
    getValue: function (dimension) {
      var data = this[hostName][dataName];
      return data.getStore().get(data.getDimensionIndex(dimension || 'value'), this.dataIndex);
    },
    // TODO: TYPE stricter type.
    setVisual: function (key, value) {
      this.dataIndex >= 0 && this[hostName][dataName].setItemVisual(this.dataIndex, key, value);
    },
    getVisual: function (key) {
      return this[hostName][dataName].getItemVisual(this.dataIndex, key);
    },
    setLayout: function (layout, merge) {
      this.dataIndex >= 0 && this[hostName][dataName].setItemLayout(this.dataIndex, layout, merge);
    },
    getLayout: function () {
      return this[hostName][dataName].getItemLayout(this.dataIndex);
    },
    getGraphicEl: function () {
      return this[hostName][dataName].getItemGraphicEl(this.dataIndex);
    },
    getRawIndex: function () {
      return this[hostName][dataName].getRawIndex(this.dataIndex);
    }
  };
}
;
;
;
zrUtil.mixin(GraphNode, createGraphDataProxyMixin('hostGraph', 'data'));
zrUtil.mixin(GraphEdge, createGraphDataProxyMixin('hostGraph', 'edgeData'));
export default Graph;
export { GraphNode, GraphEdge };