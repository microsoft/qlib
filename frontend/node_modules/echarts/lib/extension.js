
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
import { registerPreprocessor, registerProcessor, registerPostInit, registerPostUpdate, registerAction, registerCoordinateSystem, registerLayout, registerVisual, registerTransform, registerLoading, registerMap, registerUpdateLifecycle, PRIORITY } from './core/echarts.js';
import ComponentView from './view/Component.js';
import ChartView from './view/Chart.js';
import ComponentModel from './model/Component.js';
import SeriesModel from './model/Series.js';
import { isFunction, indexOf, isArray, each } from 'zrender/lib/core/util.js';
import { registerImpl } from './core/impl.js';
import { registerPainter } from 'zrender/lib/zrender.js';
var extensions = [];
var extensionRegisters = {
  registerPreprocessor: registerPreprocessor,
  registerProcessor: registerProcessor,
  registerPostInit: registerPostInit,
  registerPostUpdate: registerPostUpdate,
  registerUpdateLifecycle: registerUpdateLifecycle,
  registerAction: registerAction,
  registerCoordinateSystem: registerCoordinateSystem,
  registerLayout: registerLayout,
  registerVisual: registerVisual,
  registerTransform: registerTransform,
  registerLoading: registerLoading,
  registerMap: registerMap,
  registerImpl: registerImpl,
  PRIORITY: PRIORITY,
  ComponentModel: ComponentModel,
  ComponentView: ComponentView,
  SeriesModel: SeriesModel,
  ChartView: ChartView,
  // TODO Use ComponentModel and SeriesModel instead of Constructor
  registerComponentModel: function (ComponentModelClass) {
    ComponentModel.registerClass(ComponentModelClass);
  },
  registerComponentView: function (ComponentViewClass) {
    ComponentView.registerClass(ComponentViewClass);
  },
  registerSeriesModel: function (SeriesModelClass) {
    SeriesModel.registerClass(SeriesModelClass);
  },
  registerChartView: function (ChartViewClass) {
    ChartView.registerClass(ChartViewClass);
  },
  registerSubTypeDefaulter: function (componentType, defaulter) {
    ComponentModel.registerSubTypeDefaulter(componentType, defaulter);
  },
  registerPainter: function (painterType, PainterCtor) {
    registerPainter(painterType, PainterCtor);
  }
};
export function use(ext) {
  if (isArray(ext)) {
    // use([ChartLine, ChartBar]);
    each(ext, function (singleExt) {
      use(singleExt);
    });
    return;
  }
  if (indexOf(extensions, ext) >= 0) {
    return;
  }
  extensions.push(ext);
  if (isFunction(ext)) {
    ext = {
      install: ext
    };
  }
  ext.install(extensionRegisters);
}