import type { ECharts } from 'echarts';
import React, { PureComponent } from 'react';
import { bind, clear } from 'size-sensor';
import { pick } from './helper/pick';
import { isFunction } from './helper/is-function';
import { isString } from './helper/is-string';
import { isEqual } from './helper/is-equal';
import { EChartsReactProps, EChartsInstance } from './types';

/**
 * core component for echarts binding
 */
export default class EChartsReactCore extends PureComponent<EChartsReactProps> {
  /**
   * echarts render container
   */
  public ele: HTMLElement;

  /**
   * if this is the first time we are resizing
   */
  private isInitialResize: boolean;

  /**
   * echarts library entry
   */
  protected echarts: any;

  constructor(props: EChartsReactProps) {
    super(props);

    this.echarts = props.echarts;
    this.ele = null;
    this.isInitialResize = true;
  }

  componentDidMount() {
    this.renderNewEcharts();
  }

  // update
  componentDidUpdate(prevProps: EChartsReactProps) {
    /**
     * if shouldSetOption return false, then return, not update echarts options
     * default is true
     */
    const { shouldSetOption } = this.props;
    if (isFunction(shouldSetOption) && !shouldSetOption(prevProps, this.props)) {
      return;
    }

    // 以下属性修改的时候，需要 dispose 之后再新建
    // 1. 切换 theme 的时候
    // 2. 修改 opts 的时候
    if (!isEqual(prevProps.theme, this.props.theme) || !isEqual(prevProps.opts, this.props.opts)) {
      this.dispose();

      this.renderNewEcharts(); // 重建
      return;
    }

    // 修改 onEvent 的时候先移除历史事件再添加
    const echartsInstance = this.getEchartsInstance();
    if (!isEqual(prevProps.onEvents, this.props.onEvents)) {
      this.offEvents(echartsInstance, prevProps.onEvents);
      this.bindEvents(echartsInstance, this.props.onEvents);
    }

    // when these props are not isEqual, update echarts
    const pickKeys = ['option', 'notMerge', 'replaceMerge', 'lazyUpdate', 'showLoading', 'loadingOption'] as const;
    if (!isEqual(pick(this.props, pickKeys), pick(prevProps as any, pickKeys))) {
      this.updateEChartsOption();
    }

    /**
     * when style or class name updated, change size.
     */
    if (!isEqual(prevProps.style, this.props.style) || !isEqual(prevProps.className, this.props.className)) {
      this.resize();
    }
  }

  componentWillUnmount() {
    this.dispose();
  }

  /*
   * initialise an echarts instance
   */
  public async initEchartsInstance(): Promise<ECharts> {
    return new Promise((resolve) => {
      // create temporary echart instance
      this.echarts.init(this.ele, this.props.theme, this.props.opts);
      const echartsInstance = this.getEchartsInstance();

      echartsInstance.on('finished', () => {
        // get final width and height
        const width = this.ele.clientWidth;
        const height = this.ele.clientHeight;

        // dispose temporary echart instance
        this.echarts.dispose(this.ele);

        // recreate echart instance
        // we use final width and height only if not originally provided as opts
        const opts = {
          width,
          height,
          ...this.props.opts,
        };
        resolve(this.echarts.init(this.ele, this.props.theme, opts));
      });
    });
  }

  /**
   * return the existing echart object
   */
  public getEchartsInstance(): ECharts {
    return this.echarts.getInstanceByDom(this.ele);
  }

  /**
   * dispose echarts and clear size-sensor
   */
  private dispose() {
    if (this.ele) {
      try {
        clear(this.ele);
      } catch (e) {
        console.warn(e);
      }
      // dispose echarts instance
      this.echarts.dispose(this.ele);
    }
  }

  /**
   * render a new echarts instance
   */
  private async renderNewEcharts() {
    const { onEvents, onChartReady, autoResize = true } = this.props;

    // 1. init echarts instance
    await this.initEchartsInstance();

    // 2. update echarts instance
    const echartsInstance = this.updateEChartsOption();

    // 3. bind events
    this.bindEvents(echartsInstance, onEvents || {});

    // 4. on chart ready
    if (isFunction(onChartReady)) onChartReady(echartsInstance);

    // 5. on resize
    if (this.ele && autoResize) {
      bind(this.ele, () => {
        this.resize();
      });
    }
  }

  // bind the events
  private bindEvents(instance, events: EChartsReactProps['onEvents']) {
    function _bindEvent(eventName: string, func: Function) {
      // ignore the event config which not satisfy
      if (isString(eventName) && isFunction(func)) {
        // binding event
        instance.on(eventName, (param) => {
          func(param, instance);
        });
      }
    }

    // loop and bind
    for (const eventName in events) {
      if (Object.prototype.hasOwnProperty.call(events, eventName)) {
        _bindEvent(eventName, events[eventName]);
      }
    }
  }

  // off the events
  private offEvents(instance, events: EChartsReactProps['onEvents']) {
    if (!events) return;
    // loop and off
    for (const eventName in events) {
      if (isString(eventName)) {
        instance.off(eventName, events[eventName]);
      }
    }
  }

  /**
   * render the echarts
   */
  private updateEChartsOption(): EChartsInstance {
    const {
      option,
      notMerge = false,
      replaceMerge = null,
      lazyUpdate = false,
      showLoading,
      loadingOption = null,
    } = this.props;
    // 1. get or initial the echarts object
    const echartInstance = this.getEchartsInstance();
    // 2. set the echarts option
    echartInstance.setOption(option, { notMerge, replaceMerge, lazyUpdate });
    // 3. set loading mask
    if (showLoading) echartInstance.showLoading(loadingOption);
    else echartInstance.hideLoading();

    return echartInstance;
  }

  /**
   * resize wrapper
   */
  private resize() {
    // 1. get the echarts object
    const echartsInstance = this.getEchartsInstance();

    // 2. call echarts instance resize if not the initial resize
    // resize should not happen on first render as it will cancel initial echarts animations
    if (!this.isInitialResize) {
      try {
        echartsInstance.resize({
          width: 'auto',
          height: 'auto',
        });
      } catch (e) {
        console.warn(e);
      }
    }

    // 3. update variable for future calls
    this.isInitialResize = false;
  }

  render(): JSX.Element {
    const {
      style,
      className = '',
      echarts,
      option,
      theme,
      notMerge,
      replaceMerge,
      lazyUpdate,
      showLoading,
      loadingOption,
      opts,
      onChartReady,
      onEvents,
      shouldSetOption,
      autoResize,
      ...divHTMLAttributes
    } = this.props;
    // default height = 300
    const newStyle = { height: 300, ...style };

    return (
      <div
        ref={(e: HTMLElement) => {
          this.ele = e;
        }}
        style={newStyle}
        className={`echarts-for-react ${className}`}
        {...divHTMLAttributes}
      />
    );
  }
}
