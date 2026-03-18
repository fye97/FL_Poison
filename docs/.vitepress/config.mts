import { defineConfig } from 'vitepress'

const githubLink = 'https://github.com/fye97/FL_Poison'
const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base =
  process.env.GITHUB_ACTIONS && repoName && !repoName.endsWith('.github.io')
    ? `/${repoName}/`
    : '/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: 'FLPoison',
  description: 'FLPoison 文档站点：用户手册、配置手册与性能剖析。',
  cleanUrls: true,
  lastUpdated: true,
  head: [['meta', { name: 'theme-color', content: '#0f766e' }]],
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '用户手册', link: '/for-users' },
      { text: '配置手册', link: '/config-manual' },
      { text: '性能剖析', link: '/performance-profiling' },
      { text: 'GitHub', link: githubLink },
    ],
    sidebar: [
      {
        text: '文档',
        items: [
          { text: '首页', link: '/' },
          { text: '用户手册', link: '/for-users' },
          { text: '配置手册', link: '/config-manual' },
          { text: '性能剖析', link: '/performance-profiling' },
        ],
      },
      {
        text: '资源',
        items: [
          { text: 'Data Model PDF', link: '/datamodel.pdf' },
          { text: '框架逻辑图', link: '/#框架逻辑' },
        ],
      },
    ],
    search: {
      provider: 'local',
    },
    socialLinks: [{ icon: 'github', link: githubLink }],
    outline: 'deep',
    outlineTitle: '本页目录',
    docFooter: {
      prev: '上一页',
      next: '下一页',
    },
    sidebarMenuLabel: '文档导航',
    returnToTopLabel: '回到顶部',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    footer: {
      message: 'Released under GPL v2.',
      copyright: 'Copyright © 2026 FLPoison contributors',
    },
  },
})
