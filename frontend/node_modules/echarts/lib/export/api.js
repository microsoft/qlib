
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
// These APIs are for more advanced usages
// For example extend charts and components, creating graphic elements, formatting.
import ComponentModel from '../model/Component.js';
import ComponentView from '../view/Component.js';
import SeriesModel from '../model/Series.js';
import ChartView from '../view/Chart.js';
import SeriesData from '../data/SeriesData.js';
import * as zrender_1 from 'zrender/lib/zrender.js';
export { zrender_1 as zrender };
import * as matrix_1 from 'zrender/lib/core/matrix.js';
export { matrix_1 as matrix };
import * as vector_1 from 'zrender/lib/core/vector.js';
export { vector_1 as vector };
import * as zrUtil_1 from 'zrender/lib/core/util.js';
export { zrUtil_1 as zrUtil };
import * as color_1 from 'zrender/lib/tool/color.js';
export { color_1 as color };
export { throttle } from '../util/throttle.js';
import * as helper_1 from './api/helper.js';
export { helper_1 as helper };
export { use } from '../extension.js';
export { setPlatformAPI } from 'zrender/lib/core/platform.js';
// --------------------- Helper Methods ---------------------
export { default as parseGeoJSON } from '../coord/geo/parseGeoJson.js';
export { default as parseGeoJson } from '../coord/geo/parseGeoJson.js';
import * as number_1 from './api/number.js';
export { number_1 as number };
import * as time_1 from './api/time.js';
export { time_1 as time };
import * as graphic_1 from './api/graphic.js';
export { graphic_1 as graphic };
import * as format_1 from './api/format.js';
export { format_1 as format };
import * as util_1 from './api/util.js';
export { util_1 as util };
export { default as env } from 'zrender/lib/core/env.js';
// --------------------- Export for Extension Usage ---------------------
// export {SeriesData};
export { SeriesData as List }; // TODO: Compatitable with exists echarts-gl code
export { default as Model } from '../model/Model.js';
export { default as Axis } from '../coord/Axis.js';
export { ComponentModel, ComponentView, SeriesModel, ChartView };
// Only for GL
export { brushSingle as innerDrawElementOnCanvas } from 'zrender/lib/canvas/graphic.js';
// --------------------- Deprecated Extension Methods ---------------------
// Should use `ComponentModel.extend` or `class XXXX extend ComponentModel` to create class.
// Then use `registerComponentModel` in `install` parameter when `use` this extension. For example:
// class Bar3DModel extends ComponentModel {}
// export function install(registers) { registers.registerComponentModel(Bar3DModel); }
// echarts.use(install);
export function extendComponentModel(proto) {
  var Model = ComponentModel.extend(proto);
  ComponentModel.registerClass(Model);
  return Model;
}
export function extendComponentView(proto) {
  var View = ComponentView.extend(proto);
  ComponentView.registerClass(View);
  return View;
}
export function extendSeriesModel(proto) {
  var Model = SeriesModel.extend(proto);
  SeriesModel.registerClass(Model);
  return Model;
}
export function extendChartView(proto) {
  var View = ChartView.extend(proto);
  ChartView.registerClass(View);
  return View;
}