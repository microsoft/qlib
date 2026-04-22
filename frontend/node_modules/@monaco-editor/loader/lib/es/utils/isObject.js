function isObject(value) {
  return {}.toString.call(value).includes('Object');
}

export { isObject as default };
