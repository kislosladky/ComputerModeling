import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Загрузка данных
ds = xr.open_dataset("slp.2020.nc")

# Фильтрация по марту
march_data = ds.sel(time=ds['time'].dt.month == 3)
march_mean = march_data['slp'].mean(dim='time')

# Создание карты
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Ограничение области: Австралия
ax.set_extent([110, 155, -45, -10], crs=ccrs.PlateCarree())

# Отображение
contour = ax.contourf(ds['lon'], ds['lat'], march_mean, 60,
                      transform=ccrs.PlateCarree(), cmap='coolwarm')

# Доп. оформление
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.gridlines(draw_labels=True)

# Цветовая шкала
plt.colorbar(contour, orientation='horizontal', label='SLP (Pa)')
plt.title("Среднее давление — март (только Австралия)")
plt.show()
