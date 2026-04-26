const appJson = require("./app.json");

module.exports = ({ config }) => {
  const navApiBaseUrl =
    process.env.EXPO_PUBLIC_NAV_API_BASE_URL ||
    process.env.NAV_API_BASE_URL ||
    appJson.expo.extra?.navApiBaseUrl ||
    "";

  return {
    ...config,
    ...appJson.expo,
    extra: {
      ...(config.extra || {}),
      ...(appJson.expo.extra || {}),
      navApiBaseUrl
    }
  };
};
