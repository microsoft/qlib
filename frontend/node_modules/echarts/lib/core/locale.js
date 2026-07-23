
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
import Model from '../model/Model.js';
import env from 'zrender/lib/core/env.js';
// default import ZH and EN lang
import langEN from '../i18n/langEN.js';
import langZH from '../i18n/langZH.js';
import { isString, clone, merge } from 'zrender/lib/core/util.js';
var LOCALE_ZH = 'ZH';
var LOCALE_EN = 'EN';
var DEFAULT_LOCALE = LOCALE_EN;
var localeStorage = {};
var localeModels = {};
export var SYSTEM_LANG = !env.domSupported ? DEFAULT_LOCALE : function () {
  var langStr = (/* eslint-disable-next-line */
  document.documentElement.lang || navigator.language || navigator.browserLanguage || DEFAULT_LOCALE).toUpperCase();
  return langStr.indexOf(LOCALE_ZH) > -1 ? LOCALE_ZH : DEFAULT_LOCALE;
}();
export function registerLocale(locale, localeObj) {
  locale = locale.toUpperCase();
  localeModels[locale] = new Model(localeObj);
  localeStorage[locale] = localeObj;
}
// export function getLocale(locale: string) {
//     return localeStorage[locale];
// }
export function createLocaleObject(locale) {
  if (isString(locale)) {
    var localeObj = localeStorage[locale.toUpperCase()] || {};
    if (locale === LOCALE_ZH || locale === LOCALE_EN) {
      return clone(localeObj);
    } else {
      return merge(clone(localeObj), clone(localeStorage[DEFAULT_LOCALE]), false);
    }
  } else {
    return merge(clone(locale), clone(localeStorage[DEFAULT_LOCALE]), false);
  }
}
export function getLocaleModel(lang) {
  return localeModels[lang];
}
export function getDefaultLocaleModel() {
  return localeModels[DEFAULT_LOCALE];
}
// Default locale
registerLocale(LOCALE_EN, langEN);
registerLocale(LOCALE_ZH, langZH);