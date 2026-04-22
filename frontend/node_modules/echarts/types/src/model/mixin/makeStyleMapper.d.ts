import { PathStyleProps } from 'zrender/lib/graphic/Path.js';
import Model from '../Model.js';
export default function makeStyleMapper(properties: readonly string[][], ignoreParent?: boolean): (model: Model, excludes?: readonly string[], includes?: readonly string[]) => PathStyleProps;
