
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
export * from './export/core.js';
import { use } from './extension.js';
import { init } from './core/echarts.js';
import { install as CanvasRenderer } from './renderer/installCanvasRenderer.js';
import { install as DatasetComponent } from './component/dataset/install.js';
// Default to have canvas renderer and dataset for compitatble reason.
use([CanvasRenderer, DatasetComponent]);
// TODO: Compatitable with the following code
// import echarts from 'echarts/lib/echarts.js'
export default {
  init: function () {
    if (process.env.NODE_ENV !== 'production') {
      /* eslint-disable-next-line */
      console.error("\"import echarts from 'echarts/lib/echarts.js'\" is not supported anymore. Use \"import * as echarts from 'echarts/lib/echarts.js'\" instead;");
    }
    // @ts-ignore
    return init.apply(null, arguments);
  }
};
// Import label layout by default.
// TODO remove
import { installLabelLayout } from './label/installLabelLayout.js';
use(installLabelLayout);