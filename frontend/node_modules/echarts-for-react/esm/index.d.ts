import { EChartsReactProps, EChartsOption, EChartsInstance } from './types';
import EChartsReactCore from './core';
export type { EChartsReactProps, EChartsOption, EChartsInstance };
export default class EChartsReact extends EChartsReactCore {
    constructor(props: EChartsReactProps);
}
